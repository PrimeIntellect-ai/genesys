import re
import sympy
from sympy.parsing.latex import parse_latex
from typing import Optional, Dict
from genesys.verifiers.base_verifier import BaseVerifier

SUBSTITUTIONS = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]

REMOVED_EXPRESSIONS = [
    "square",
    "ways",
    "integers",
    "dollars",
    "mph",
    "inches",
    "ft",
    "hours",
    "km",
    "units",
    "\\ldots",
    "sue",
    "points",
    "feet",
    "minutes",
    "digits",
    "cents",
    "degrees",
    "cm",
    "gm",
    "pounds",
    "meters",
    "meals",
    "edges",
    "students",
    "childrentickets",
    "multiples",
    "\\text{s}",
    "\\text{.}",
    "\\text{\ns}",
    "\\text{}^2",
    "\\text{}^3",
    "\\text{\n}",
    "\\text{}",
    r"\mathrm{th}",
    r"^\circ",
    r"^{\circ}",
    r"\;",
    r",\!",
    "{,}",
    '"',
    "\\dots",
]


class MathVerifier(BaseVerifier):
    max_parallel = 10
    timeout = 20

    def verify(self, result: Dict) -> int:
        """
        Given a result dict containing:
          - "llm_response": the raw model output (string)
          - "verification_info": a dict containing at least "ground_truth" (the expected answer)
        this method extracts candidate answers from the model output, normalizes them,
        and then compares against the ground truth.

        Returns 1 if any candidate answer matches (according to our equivalence functions)
        and 0 otherwise.
        """
        model_output = result["llm_response"]
        ground_truth_answer = result["verification_info"]["ground_truth"]

        raw_answer = model_output
        all_answers = []

        boxed_answer = self._last_boxed_only_string(raw_answer)
        if boxed_answer is not None:
            try:
                boxed_answer = self._remove_boxed(boxed_answer)
            except AssertionError:
                boxed_answer = None
        if boxed_answer is not None:
            all_answers.append(boxed_answer)

        minerva_answer = self._normalize_final_answer(self._get_unnormalized_answer(raw_answer))
        if minerva_answer is not None and minerva_answer != "[invalidanswer]":
            all_answers.append(minerva_answer)

        if len(all_answers) == 0:
            dollars = [m.start() for m in re.finditer(r"\$", raw_answer)]
            if len(dollars) > 1:
                answer = self._normalize_final_answer(raw_answer[dollars[-2] + 1 : dollars[-1]])
                all_answers.append(answer)

        if len(all_answers) == 0:
            all_answers.append(self._normalize_final_answer(model_output))

        matched = False
        for answer in all_answers:
            if self._is_equiv(answer, ground_truth_answer):
                matched = True
                break
            elif self._hendrycks_is_equiv(answer, ground_truth_answer):
                matched = True
                break

        return dict(score=int(matched), verification_result_info=dict())

    def _last_boxed_only_string(self, string: str) -> Optional[str]:
        idx = string.rfind("\\boxed")
        if "\\boxed " in string:
            return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
        if idx < 0:
            idx = string.rfind("\\fbox")
            if idx < 0:
                return None

        i = idx
        right_brace_idx = None
        num_left_braces_open = 0
        while i < len(string):
            if string[i] == "{":
                num_left_braces_open += 1
            if string[i] == "}":
                num_left_braces_open -= 1
                if num_left_braces_open == 0:
                    right_brace_idx = i
                    break
            i += 1

        if right_brace_idx is None:
            retval = None
        else:
            retval = string[idx : right_brace_idx + 1]

        return retval

    def _remove_boxed(self, s: str) -> str:
        if "\\boxed " in s:
            left = "\\boxed "
            assert s[: len(left)] == left
            return s[len(left) :]
        left = "\\boxed{"
        assert s[: len(left)] == left
        assert s[-1] == "}"
        return s[len(left) : -1]

    def _get_unnormalized_answer(self, text: str) -> str:
        INVALID_ANSWER = "[invalidanswer]"
        end_seq = "I hope it is correct."
        text += end_seq
        match = re.search(
            r"Final Answer: The final answer is(.*?). I hope it is correct.",
            text,
        )
        if match:
            return match.group(1).strip()
        else:
            return INVALID_ANSWER

    def _normalize_final_answer(self, final_answer: str) -> str:
        final_answer = final_answer.split("=")[-1]
        for before, after in SUBSTITUTIONS:
            final_answer = final_answer.replace(before, after)
        for expr in REMOVED_EXPRESSIONS:
            final_answer = final_answer.replace(expr, "")
        final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", r"$\3$", final_answer)
        final_answer = re.sub(r"(\\text\{)(.*?)(\})", r"\2", final_answer)
        final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", r"\2", final_answer)
        final_answer = re.sub(r"(\\overline\{)(.*?)(\})", r"\2", final_answer)
        final_answer = re.sub(r"(\\boxed\{)(.*)(\})", r"\2", final_answer)
        final_answer = re.sub(r"(frac)([^{])(.)", r"frac{\2}{\3}", final_answer)
        final_answer = re.sub(r"(sqrt)([^{])", r"sqrt{\2}", final_answer)
        final_answer = final_answer.replace("$", "")
        if final_answer.replace(",", "").isdigit():
            final_answer = final_answer.replace(",", "")
        return final_answer

    def _is_equiv(self, x1: str, x2: str) -> bool:
        try:
            try:
                parsed_x1 = parse_latex(x1)
                parsed_x2 = parse_latex(x2)
            except (sympy.parsing.latex.errors.LaTeXParsingError, sympy.SympifyError, TypeError):
                print(f"couldn't parse one of {x1} or {x2}")
                return False
            try:
                diff = parsed_x1 - parsed_x2
            except TypeError:
                print(f"couldn't subtract {x1} and {x2}")
                return False
            try:
                if sympy.simplify(diff) == 0:
                    return True
                else:
                    return False
            except ValueError:
                print(f"Had some trouble simplifying when comparing {x1} and {x2}")
        except ImportError as e:
            print(e)
            raise
        except Exception as e:
            print(f"Failed comparing {x1} and {x2} with {e}")
            return False

    def _fix_fracs(self, string: str) -> str:
        substrs = string.split("\\frac")
        new_str = substrs[0]
        if len(substrs) > 1:
            substrs = substrs[1:]
            for substr in substrs:
                new_str += "\\frac"
                if substr[0] == "{":
                    new_str += substr
                else:
                    try:
                        assert len(substr) >= 2
                    except AssertionError:
                        return string
                    a = substr[0]
                    b = substr[1]
                    if b != "{":
                        if len(substr) > 2:
                            post_substr = substr[2:]
                            new_str += "{" + a + "}{" + b + "}" + post_substr
                        else:
                            new_str += "{" + a + "}{" + b + "}"
                    else:
                        if len(substr) > 2:
                            post_substr = substr[2:]
                            new_str += "{" + a + "}" + b + post_substr
                        else:
                            new_str += "{" + a + "}" + b
        string = new_str
        return string

    def _fix_a_slash_b(self, string: str) -> str:
        if len(string.split("/")) != 2:
            return string
        a = string.split("/")[0]
        b = string.split("/")[1]
        try:
            a = int(a)
            b = int(b)
            assert string == "{}/{}".format(a, b)
            new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
            return new_string
        except AssertionError:
            return string

    def _remove_right_units(self, string: str) -> str:
        if "\\text{ " in string:
            splits = string.split("\\text{ ")
            assert len(splits) == 2
            return splits[0]
        else:
            return string

    def _fix_sqrt(self, string: str) -> str:
        if "\\sqrt" not in string:
            return string
        splits = string.split("\\sqrt")
        new_string = splits[0]
        for split in splits[1:]:
            if split[0] != "{":
                a = split[0]
                new_substr = "\\sqrt{" + a + "}" + split[1:]
            else:
                new_substr = "\\sqrt" + split
            new_string += new_substr
        return new_string

    def _strip_string(self, string: str) -> str:
        string = string.replace("\n", "")
        string = string.replace("\\!", "")
        string = string.replace("\\\\", "\\")
        string = string.replace("tfrac", "frac")
        string = string.replace("dfrac", "frac")
        string = string.replace("\\left", "")
        string = string.replace("\\right", "")
        string = string.replace("^{\\circ}", "")
        string = string.replace("^\\circ", "")
        string = string.replace("\\$", "")
        string = self._remove_right_units(string)
        string = string.replace("\\%", "")
        string = string.replace("\%", "")
        string = string.replace(" .", " 0.")
        string = string.replace("{.", "{0.")
        if len(string) == 0:
            return string
        if string[0] == ".":
            string = "0" + string
        if len(string.split("=")) == 2:
            if len(string.split("=")[0]) <= 2:
                string = string.split("=")[1]
        string = self._fix_sqrt(string)
        string = string.replace(" ", "")
        string = self._fix_fracs(string)
        if string == "0.5":
            string = "\\frac{1}{2}"
        string = self._fix_a_slash_b(string)
        return string

    def _hendrycks_is_equiv(self, str1: str, str2: str, verbose=False) -> bool:
        if str1 is None and str2 is None:
            print("WARNING: Both None")
            return True
        if str1 is None or str2 is None:
            return False
        try:
            ss1 = self._strip_string(str1)
            ss2 = self._strip_string(str2)
            if verbose:
                print(ss1, ss2)
            return ss1 == ss2
        except Exception:
            return str1 == str2

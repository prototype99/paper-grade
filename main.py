import json
import language_tool_python
import textstat
from dotenv import load_dotenv
from deertick.src.py.agent import Agent

# Load environment variables from .env file
load_dotenv()


class AIPaperGrader:
    """
    An AI-powered paper grader that uses a hybrid approach of objective metrics
    and subjective LLM-based evaluation against a rubric.
    """

    def __init__(
            self,
            model="OpenAI: GPT-4 Turbo",
    ):
        """
        Initializes the grader with an OpenAI client and a grammar tool.
        """
        self.agent = Agent(
            model,
            "You are an expert AI teaching assistant. Your task is to grade a student's paper based on a given assignment prompt and a detailed grading rubric. You must provide a score and constructive feedback for EACH criterion in the rubric. Your analysis must be objective and strictly adhere to the rubric's definitions.",
            "openai"
        )

        # Initialize the grammar tool (it may download language data on first run)
        try:
            self.lang_tool = language_tool_python.LanguageTool(
                'en-US'
            )
        except Exception as e:
            print(
                f"Could not initialize language tool. Grammar checks will be skipped. Error: {e}"
            )
            self.lang_tool = None

    def _check_objective_metrics(
            self,
            paper_text: str
    ) -> dict:
        """
        Calculates objective metrics for the paper.
        """
        word_count = len(
            paper_text
            .split()
        )
        readability_score = textstat.flesch_reading_ease(
            paper_text
        )

        grammar_errors = 0
        if self.lang_tool:
            matches = self.lang_tool.check(
                paper_text
            )
            grammar_errors = len(
                matches
            )

        return {
            "word_count": word_count,
            "readability_score_flesch": readability_score,
            "grammar_and_spelling_errors": grammar_errors
        }

    def _evaluate_with_llm(
            self,
            assignment_prompt: str,
            rubric: dict,
            student_paper: str
    ) -> dict:
        """
        Uses the LLM to evaluate the paper against the rubric.
        """
        # We will format the rubric nicely for the prompt
        rubric_str = json.dumps(
            rubric,
            indent=2
        )

        user_prompt = f"""
        Please grade the following student paper.

        [ASSIGNMENT PROMPT]:
        {assignment_prompt}

        [GRADING RUBRIC]:
        {rubric_str}

        [STUDENT SUBMISSION]:
        {student_paper}

        [YOUR TASK]:
        Evaluate the student's submission based STRICTLY on the provided rubric. For each criterion,
        provide a numerical score and detailed, constructive feedback explaining why you gave that score.
        Your feedback should be helpful and guide the student on how to improve.

        Return your entire response as a single, valid JSON object. The JSON object should have two main keys:
        1. "summary_feedback": A brief overall summary of the paper's strengths and weaknesses.
        2. "criteria_breakdown": A list of objects, where each object represents a criterion from the rubric and contains:
           - "criterion": The name of the criterion (e.g., "Clarity and Argument").
           - "score": The score you awarded for this criterion.
           - "max_score": The maximum possible score for this criterion.
           - "feedback": Your detailed feedback for this specific criterion.

        Do not include any text outside of the JSON object.
        """

        try:
            response = self.agent.generate_response(
                self.agent.system_prompt,
                f"""
                        Please grade the following student paper.

                        [ASSIGNMENT PROMPT]:
                        {assignment_prompt}

                        [GRADING RUBRIC]:
                        {rubric_str}

                        [STUDENT SUBMISSION]:
                        {student_paper}

                        [YOUR TASK]:
                        Evaluate the student's submission based STRICTLY on the provided rubric. For each criterion,
                        provide a numerical score and detailed, constructive feedback explaining why you gave that score.
                        Your feedback should be helpful and guide the student on how to improve.

                        Return your entire response as a single, valid JSON object. The JSON object should have two main keys:
                        1. "summary_feedback": A brief overall summary of the paper's strengths and weaknesses.
                        2. "criteria_breakdown": A list of objects, where each object represents a criterion from the rubric and contains:
                           - "criterion": The name of the criterion (e.g., "Clarity and Argument").
                           - "score": The score you awarded for this criterion.
                           - "max_score": The maximum possible score for this criterion.
                           - "feedback": Your detailed feedback for this specific criterion.

                        Do not include any text outside of the JSON object.
                """
            )
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ],
                # This ensures the model outputs valid JSON
                response_format={
                    "type": "json_object"
                },
                temperature=0.2  # Lower temperature for more consistent, objective grading
            )

            llm_evaluation = json.loads(
                response.choices[0]
                .message.content
            )
            return llm_evaluation

        except Exception as e:
            print(
                f"An error occurred during LLM evaluation: {e}"
            )
            return {
                "summary_feedback": "Error: Could not get evaluation from the AI model.",
                "criteria_breakdown": []
            }

    def grade_paper(
            self,
            assignment_prompt: str,
            rubric: dict,
            student_paper: str
    ) -> dict:
        """
        Orchestrates the entire grading process.
        """
        print(
            "1. Analyzing objective metrics..."
        )
        objective_results = self._check_objective_metrics(
            student_paper
        )

        print(
            "2. Performing subjective evaluation with AI..."
        )
        llm_results = self._evaluate_with_llm(
            assignment_prompt,
            rubric,
            student_paper
        )

        # Calculate total score from the LLM's breakdown
        total_score = sum(item.get('score', 0) for item in llm_results.get('criteria_breakdown', []))
        max_total_score = sum(item.get('max_score', 0) for item in rubric.get('criteria', []))

        print(
            "3. Compiling final report..."
        )
        final_grade = {
            "overall_score": total_score,
            "max_possible_score": max_total_score,
            "summary_feedback": llm_results.get(
                'summary_feedback'
            ),
            "criteria_breakdown": llm_results.get(
                'criteria_breakdown'
            ),
            "objective_metrics": objective_results
        }

        return final_grade

    @staticmethod
    def print_report(
            grade_report: dict
    ):
        """
        Prints the final grade report in a readable format.
        """
        print("\n" + "=" * 50)
        print(" " * 15 + "AI GRADING REPORT")
        print("=" * 50 + "\n")

        print(f"FINAL SCORE: {grade_report['overall_score']} / {grade_report['max_possible_score']}\n")

        print("--- Overall Summary ---")
        print(grade_report.get('summary_feedback', 'N/A'))
        print("\n" + "-" * 25 + "\n")

        print("--- Detailed Breakdown by Criterion ---")
        for item in grade_report.get('criteria_breakdown', []):
            print(f"\n[{item['criterion']}]")
            print(f"  Score: {item['score']} / {item['max_score']}")
            print(f"  Feedback: {item['feedback']}")

        print("\n" + "-" * 25 + "\n")

        print("--- Objective Metrics ---")
        metrics = grade_report.get('objective_metrics', {})
        print(f"  Word Count: {metrics.get('word_count')}")
        print(f"  Flesch Reading Ease: {metrics.get('readability_score_flesch'):.2f} (Higher is easier to read)")
        print(f"  Grammar & Spelling Issues Found: {metrics.get('grammar_and_spelling_errors')}")
        print("\n" + "=" * 50)


if __name__ == '__main__':
    # 1. DEFINE THE ASSIGNMENT PROMPT
    ASSIGNMENT_PROMPT = """
    Write a 300-500 word essay on the role of technology in modern education. 
    Discuss both the advantages and disadvantages, and provide at least one specific example 
    for each. Conclude with your own perspective on the future of educational technology.
    """

    # 2. DEFINE THE GRADING RUBRIC
    # This is the most important part. A detailed rubric gets better results.
    GRADING_RUBRIC = {
        "criteria": [
            {
                "criterion": "Argument and Analysis",
                "max_score": 40,
                "description": "Evaluates the clarity, depth, and coherence of the argument. Assesses discussion of both advantages and disadvantages."
            },
            {
                "criterion": "Use of Evidence",
                "max_score": 30,
                "description": "Checks if the student provided specific, relevant examples for both advantages and disadvantages as required."
            },
            {
                "criterion": "Structure and Organization",
                "max_score": 20,
                "description": "Assesses the logical flow of the essay, including introduction, body paragraphs, and conclusion."
            },
            {
                "criterion": "Clarity and Mechanics",
                "max_score": 10,
                "description": "Evaluates grammar, spelling, punctuation, and overall readability. Adherence to word count."
            }
        ]
    }

    # 3. A SAMPLE STUDENT PAPER
    # This paper is intentionally written to be good but not perfect.
    STUDENT_PAPER = """
    Technology in Education Today

    The role of technology in education has become super important. It has changed how students learn and teachers teach. There are many good things about it, but also some bad things.

    One major advantage is access to information. With the internet, students can find anything they want instantly. For example, a student studying history can watch documentaries or read original documents online, which is much better than just a textbook. This makes learning more engaging. Digital tools like interactive whiteboards and educational apps also make lessons more fun.

    However, there are also disadvantages. The biggest one is the digital divide, where not all students have equal access to computers or reliable internet at home. This creates inequality. Another problem is the potential for distraction. When students use laptops in class, they might be tempted to go on social media instead of paying attention to the lesson. It's a real issue for classroom management.

    In conclusion, I believe that technology is a powerful tool for education. The benefits, like access to resources, are huge. We need to work on solving the problems like the digital divide to make sure everyone can benefit. The future of education will definitely involve more technology, and it will probably be even more integrated into learning.
    """

    # 4. RUN THE GRADER
    try:
        grader = AIPaperGrader()
        final_grade_report = grader.grade_paper(ASSIGNMENT_PROMPT, GRADING_RUBRIC, STUDENT_PAPER)

        # 5. PRINT THE FINAL REPORT
        AIPaperGrader.print_report(final_grade_report)

    except ValueError as e:
        print(e)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

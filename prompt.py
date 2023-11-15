# from datasets import load_dataset
# dataset = load_dataset("truthful_qa", 'generation')
from datasets import load_dataset, load_metric, Dataset
import pandas as pd
import openai
import pickle, json
import copy
import sys

# import bleurt
sys.path.insert(0, 'bleurt')
from evaluator import Evaluator

OPENAI_API_KEY = "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"
# MODEL = "gpt-4"
PROCESS_PROMPT_MODEL = "gpt-4"
MODEL = "gpt-3.5-turbo"

SYSTEM_PROMPT_PROCESSING_ROLE = ""


class GPT:
    def __init__(self, model="gpt-3.5-turbo"):
        self.model = model
        self.query = SYSTEM_PROMPT_PROCESSING_ROLE
        openai.api_key = OPENAI_API_KEY

        existing_prompt_output_dataset = open('prompt_output_dataset.json')
        data = json.load(existing_prompt_output_dataset)
        self.existing_questions = data
        
    
    def checkin_existing_question(self,q):
        for instance in self.existing_questions:
            if instance['question'] == q:
                return True
        return False

    def generate_distractor(self):

        train_dataset = load_dataset("json", data_files="train_dataset.json")
        test_dataset = load_dataset("json", data_files="test_dataset.json")

        
        train_categories_for_QA = {}
        for train_instance in train_dataset['train']:
            category = train_instance['category']
            
            if category not in train_categories_for_QA:
                train_categories_for_QA[category] = []
            
            temp_instance = {
                "question": train_instance['question'], 
                "distactors": train_instance['incorrect_answers']
            }

            train_categories_for_QA[category].append(temp_instance)

        # print(train_categories_for_QA)

        i=0
        # output_test = []
        for index , test_instance in enumerate(test_dataset['train']):
            category = test_instance['category']
            examples = ""

            if self.checkin_existing_question(test_instance['question']):
                continue

            for instance in train_categories_for_QA[category]:
                examples += "Question: " + instance['question'] + " and its distractors: " + str(instance['distactors']) + ". "         

            target_question = test_instance['question']
            target_correct_answer = test_instance['best_answer']
            system_content = "You are a distractor generation helper for generating distractors for multiple-choice questions. You will learn distracotr from the several examples user provides to you and generate three distractors for a multiple-choice question. You should answer one sentence and divid each distractor with a semicolon. For instance, these are three distractors you generated, and you should organize them in this way: Ireland is part of Great Britain because it was historically colonized.; Ireland is part of Great Britain due to the influence of colonialism.; One reason Ireland is part of Great Britain is because of historical colonization."

            user_prompt = f"Here are some exmamples of question, correct answer, and distractors: {examples}. Please generate distractor for this question {target_question}, which we already have its correct answer {target_correct_answer}"
            completion = openai.ChatCompletion.create(
                model=MODEL,
                    messages=[
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": user_prompt}
                    ]
                )
            output = completion.choices[0].message.content


            distractors = output.split(';')
            distractors = [distractor.strip() for distractor in distractors]
            
            
            if len(distractors) >0:
                print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
                print(output)
                self.existing_questions.append({"category": category, "question":test_instance['question'], "distractors":distractors, "best_answer": test_instance['best_answer'], "correct_answers":test_instance['correct_answers'], "incorrect_answers": test_instance['incorrect_answers']})

            json_string = json.dumps(self.existing_questions, indent=4)
            with open("prompt_output_dataset.json", "w") as outfile:
                outfile.write(json_string)

            i+=1

        return 

    def refine_prompt_output(self):
        train_dataset = load_dataset("json", data_files="train_dataset.json")
        test_dataset = load_dataset("json", data_files="test_dataset.json")
        train_categories_for_QA = {}
        for train_instance in train_dataset['train']:
            category = train_instance['category']
            
            if category not in train_categories_for_QA:
                train_categories_for_QA[category] = []
            
            temp_instance = {
                "question": train_instance['question'], 
                "distactors": train_instance['incorrect_answers']
            }

            train_categories_for_QA[category].append(temp_instance)

        for index,test_instance in enumerate(self.existing_questions):
            category = test_instance['category']
            examples = ""

            for instance in train_categories_for_QA[category]:
                examples += "Question: " + instance['question'] + " and its distractors: " + str(instance['distactors']) + ". "         


            if len(test_instance['distractors']) != 3:
                print("$$$$$$$$$$$$$ found one!")
                target_question = test_instance['question']
                target_correct_answer = test_instance['best_answer']
                system_content = "You are a distractor generation helper for generating distractors for multiple-choice questions. You will learn distracotr from the several examples user provides to you and generate three distractors for a multiple-choice question. You should answer one sentence and divid each distractor with a semicolon. For instance, these are three distractors you generated, and you should organize them in this way: Ireland is part of Great Britain because it was historically colonized.; Ireland is part of Great Britain due to the influence of colonialism.; One reason Ireland is part of Great Britain is because of historical colonization."

                user_prompt = f"Here are some exmamples of question, correct answer, and distractors: {examples}. Please generate distractor for this question {target_question}, which we already have its correct answer {target_correct_answer}"
                completion = openai.ChatCompletion.create(
                    model=MODEL,
                        messages=[
                            {"role": "system", "content": system_content},
                            {"role": "user", "content": user_prompt}
                        ]
                    )
                output = completion.choices[0].message.content


                distractors = output.split(';')
                distractors = [distractor.strip() for distractor in distractors]
                
                print(len(distractors), distractors)
                self.existing_questions[index]['distractors'] = distractors

            json_string = json.dumps(self.existing_questions, indent=4)
            with open("prompt_output_dataset.json", "w") as outfile:
                outfile.write(json_string)
                
                
    def evaluate(self):
        evaluator = Evaluator()
        
        for index, instance in enumerate(self.existing_questions):
            distractors = instance['distractors']
            incorrect_answers = instance['incorrect_answers']
            scores = []
            score_correspondences = []
            print("distractors", len(distractors))
            for distractor in distractors:
                references = incorrect_answers
                candidates = [distractor for i in range(len(incorrect_answers))]
                output_scores = evaluator.evaluate(references, candidates)
                max_score = max(output_scores)
                score_index = output_scores.index(max_score)
                scores.append(max_score)
                score_correspondence = references[score_index]
                score_correspondences.append(score_correspondence)
                # print(scores, max(scores))

            self.existing_questions[index]['scores'] = scores
            self.existing_questions[index]['score_correspondences'] = score_correspondences
            print(scores, score_correspondences)
            
            json_string = json.dumps(self.existing_questions, indent=4)
            with open("prompt_output_dataset.json", "w") as outfile:
                outfile.write(json_string)
        return

if __name__ == "__main__":
    gpt = GPT('gpt-4')
    gpt.evaluate()
    # gpt.refine_prompt_output()
    # gpt.generate_distractor()
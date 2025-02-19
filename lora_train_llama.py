import torch

import pandas as pd
from tqdm import tqdm

tqdm.pandas()

import argparse
import os

from ast import literal_eval

from datasets import load_dataset, DatasetDict, Dataset, load_from_disk, concatenate_datasets

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    pipeline
)
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

import argparse

def prepare_datasets(unknown_rate, high_known_rate=1, parahprase_flag=True):
    def clean_out_ans(ans_dict):
        ans = ans_dict['answer'][0]
        ans_dict['answer'] = {'aliases': ans['aliases'],
                              'normalized_aliases': ans['normalized_aliases']}
        return ans_dict
    # Load datasets from disk
    examples_unk = load_from_disk(args.data_path)['full']
    #examples_unk = load_from_disk(r"./dataset/full_dbpedia_21k_with_3k_para")['full']
    #examples_unk = load_from_disk(r"./dataset/mistral_dataset_full_dbpedia_v2")['full'] #only for mistral test
    
    # Filter Unknown data based on the required conditions
    if unknown_rate <= 500:
        examples_unk = examples_unk.filter(lambda x: x['Category'] == 'Unknown' and x['is_para'] and x['para_len'] == 100)
    else:
        examples_unk = examples_unk.filter(lambda x: x['Category'] == 'Unknown' and x['is_para'])

    # Split examples for testing
    examples_unk_test = examples_unk.select(range(min(unknown_rate, len(examples_unk))))
    examples_unk_test = examples_unk_test.map(clean_out_ans)
    
    # Prepare additional data for training
    if not parahprase_flag:
        #examples_hk = load_from_disk(r"./dataset/UNK_HN_dataset_90k")['full']
        #examples_hk = load_from_disk(r"./dataset/mistral_dataset_full_UNK_HN_90k")['full'] #only for mistral test
        examples_hk = load_from_disk(args.data_path)['full']
        examples_hk = examples_hk.filter(lambda x: x['Category'] == 'HighlyKnown')
        examples_hk = examples_hk.select(range(min(high_known_rate * unknown_rate, len(examples_hk))))
        examples_hk = examples_hk.map(clean_out_ans)
        examples_unk_train = concatenate_datasets([examples_hk, 
                                                   examples_unk_test
                                                    ])
    else:
        examples_para = examples_unk_test.map(lambda x: {'question': x['para'][:high_known_rate]}, remove_columns=["para"] )
        examples_para = Dataset.from_pandas(pd.DataFrame(examples_para).explode('question'))
        examples_para = examples_para.select(range(0)) if high_known_rate <1 else examples_para
        examples_unk_train = concatenate_datasets([examples_para,
                                                    examples_unk_test
                                                    ])
    
    dataset = DatasetDict({
        'train': examples_unk_train,
        'test': examples_unk_test,
        'valid': examples_unk_test,
    })
    
    return dataset

def prompt_system_generator(dataset):
    end_text = []
    for question_text, answer_text in zip(*dataset):
        end_text.extend([{"role": "user", "content": "Question: " + question_text}])
        end_text.extend(
            [
                {
                    "role": "assistant",
                    "content": "Answer: " + answer_text["normalized_aliases"][0],
                }
            ]
        )
    return end_text

def main(path, unknown, high_known, rank, paraphrase, seed):

    model_path = path
    high_known_rate = high_known
    unknown_rate = unknown
    lora_rank = rank
    parahprase_flag = paraphrase
    

    N_SHOT_test = 4
    N_EXP_test = 10
    N_SHOT_train = 0
    N_EXP_train = 1
    LR = 1e-3
    BS = 8  # batch size, orig 32
    EPOCHS = 10

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        #padding_side="left",
        trust_remote_code=True,
    )
    quant_config = BitsAndBytesConfig(
        load_in_8bit=True,
        #bnb_4bit_quant_type="nf4",
        #bnb_4bit_use_double_quant=True,
        #bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        #quantization_config=quant_config,
        #attn_implementation="flash_attention_2",
        attn_implementation="eager",
    )

    config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank * 2,
        use_rslora=True,
        lora_dropout=0.1,
        bias="none",
        # target_modules="all-linear",
        target_modules=["down_proj", "gate_proj", "up_proj"],
        task_type="CAUSAL_LM",
    )

    #model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, config)

    dataset_fewshot = load_dataset("trivia_qa", "rc", trust_remote_code=True)

    tokenizer.pad_token_id = tokenizer.eos_token_id

    def primer_system_user(examples, N_EXP, N_SHOT, eval_mode):
        prompted_text = "Answer the following question."
        outputs = []

        # Precompute and reuse constant data
        questions = examples["question"]
        answers = [examples["answer"][0]]

        # Precompute primed_data
        primed_data = {
            "question": questions * N_EXP,
            "answer": answers * N_EXP,
        }

        # Preselect columns once
        train_dataset = dataset_fewshot["train"].select_columns(["question", "answer"])

        # Precompute messages template for efficiency
        system_message = {"role": "system", "content": prompted_text}

        # Loop over add_iter once, calculating dataset_to_eval once per iteration
        for add_iter in range(N_EXP):
            start_index = N_SHOT * add_iter
            end_index = N_SHOT * (1 + add_iter)

            # Precompute dataset to evaluate
            dataset_to_eval = train_dataset[start_index:end_index].values()

            # Generate message list for each question in one go using list comprehension
            outputs.extend(
                [
                    tokenizer.apply_chat_template(
                        [system_message]
                        + prompt_system_generator(dataset_to_eval)
                        + [
                            {"role": "user", "content": f"Question: {question}"},
                        ]
                        + [
                            {
                                "role": "assistant",
                                "content": f"Answer: {answers[0]['aliases'][0]}",
                            }
                        ]
                        * (1 - int(eval_mode)),  # BAD MANNER
                        tokenize=False,
                        add_generation_prompt=eval_mode,
                    )
                    for question in questions
                ]
            )

        primed_data["primed_question"] = outputs

        return primed_data

    dataset = prepare_datasets(unknown_rate, high_known_rate, parahprase_flag)

    if parahprase_flag:
        print(f'Used {high_known_rate} Paraphrases for {unknown_rate} Unknown train')
    else:
        print(f'Used {high_known_rate} HighKnown for {unknown_rate} Unknown train')
    print(dataset)

    from functools import partial

    dataset["train"] = dataset["train"].map(
        partial(
            primer_system_user, N_EXP=N_EXP_train, N_SHOT=N_SHOT_train, eval_mode=False
        ),
        batched=True,
        remove_columns=dataset["train"].column_names,
        batch_size=1,
    )

    dataset["test"] = dataset["test"].map(
        partial(
            primer_system_user, N_EXP=N_EXP_test, N_SHOT=N_SHOT_test, eval_mode=True
        ),
        batched=True,
        remove_columns=dataset["test"].column_names,
        batch_size=1,
    )

    dataset["valid"] = dataset["valid"].map(
        partial(
            primer_system_user, N_EXP=N_EXP_test, N_SHOT=N_SHOT_test, eval_mode=True
        ),
        batched=True,
        remove_columns=dataset["valid"].column_names,
        batch_size=1,
    )

    def tokenize(element):
        return tokenizer(
            element["primed_question"],
            truncation=True,
            max_length=768,
            add_special_tokens=False,
        )

    dataset_tokenized = dataset.map(
        tokenize,
        batched=True,
        num_proc=os.cpu_count(),  # multithreaded
        # remove_columns=["question","answer","text"]     # don't need the strings anymore, we have tokens from here on
    )

    def collate(elements):
        tokenlist = [e["input_ids"] for e in elements]
        tokens_maxlen = max([len(t) for t in tokenlist])  # length of longest input

        input_ids, labels, attention_masks = [], [], []
        for tokens in tokenlist:
            # how many pad tokens to add for this sample
            pad_len = tokens_maxlen - len(tokens)

            # pad input_ids with pad_token, labels with ignore_index (-100) and set attention_mask 1 where content, otherwise 0
            input_ids.append(tokens + [tokenizer.pad_token_id] * pad_len)
            labels.append(tokens + [-100] * pad_len)
            attention_masks.append([1] * len(tokens) + [0] * pad_len)

        batch = {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(attention_masks),
        }
        return batch

    flag_str = 'Paraphrase' if parahprase_flag else 'HighKnown'
    args = TrainingArguments(
        run_name=f"llama3_1_8b_instr_lora{lora_rank}_bs32_trained_on_{unknown_rate}Unknown_{high_known_rate}Rephrase",
        output_dir=f"../lora_r1a1/{flag_str}/lora{lora_rank}_onlyproj_bs{BS}_LR{LR}_seed{seed}_trained_on_{unknown_rate}Unknown_{high_known_rate}{flag_str}",
        per_device_train_batch_size=BS,
        per_device_eval_batch_size=128,
        eval_strategy="no",
        logging_steps=100,
        save_steps=0,
        save_strategy="epoch",
        save_total_limit=20,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        group_by_length=True,
        report_to="none",
        seed=seed,
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=collate,
        train_dataset=dataset_tokenized["train"],
        eval_dataset=dataset_tokenized["test"],
        args=args,
    )

    trainer.train()
    trainer.save_model()

    dataset.save_to_disk(
        os.path.join(trainer.args.output_dir, "dataset_to_train.dataset")
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some variables.")
    parser.add_argument("--path", type=str, help="The model path")
    parser.add_argument("--unknown", type=int, help="N of unknown", required=True)
    parser.add_argument("--high_known", type=int, help="N of HighlyKnown", required=True)
    parser.add_argument("--rank", type=int, help="LoRA rank", required=True)
    parser.add_argument("--paraphrase", action='store_true', help="Paraphrasse or HighKnown?")
    parser.add_argument("--seed", type=int, help="seed")
    parser.add_argument("--data_path", type=str, help="Path to training data", required=True)

    args = parser.parse_args()
    main(args.path, args.unknown, args.high_known, args.rank, args.paraphrase, args.seed)

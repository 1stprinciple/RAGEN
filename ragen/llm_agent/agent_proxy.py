import argparse
import json
import os
import time
from typing import Dict, List

import hydra
from transformers import AutoModelForCausalLM, AutoTokenizer

from verl import DataProto

from .base_llm import ConcurrentLLM
from .ctx_manager import ContextManager
from .es_manager import EnvStateManager


class ApiCallingWrapperWg:
    """Wrapper class for API-based LLM calls that fits into the VERL framework"""

    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        model_info = config.model_info[config.model_config.model_name]
        self.llm_kwargs = model_info.generation_kwargs


        self.llm = ConcurrentLLM(
			provider=model_info.provider_name,
            model_name=model_info.model_name,
            max_concurrency=config.model_config.max_concurrency
        )

        print(f'API-based LLM ({model_info.provider_name} - {model_info.model_name}) initialized')


    def generate_sequences(self, lm_inputs: DataProto) -> DataProto:
        """
        Convert the input ids to text, make API calls to generate responses,
        and create a DataProto with the results.
        """

        messages_list = lm_inputs.non_tensor_batch['messages_list'].tolist()
        results, failed_messages = self.llm.run_batch(
            messages_list=messages_list,
            **self.llm_kwargs
        )
        assert not failed_messages, f"Failed to generate responses for the following messages: {failed_messages}"

        texts = [result["response"] for result in results]
        # print(f'[DEBUG] texts: {texts}')
        lm_outputs = DataProto()
        lm_outputs.non_tensor_batch = {
			'response_texts': texts,
			'env_ids': lm_inputs.non_tensor_batch['env_ids'],
			'group_ids': lm_inputs.non_tensor_batch['group_ids']
		} # this is a bit hard-coded to bypass the __init__ check in DataProto
        lm_outputs.meta_info = lm_inputs.meta_info

        return lm_outputs

class LLMAgentProxy:
	"""
	The proxy means the llm agent is trying to generate some rollout **at this time**, **at this model state**, **at this env state from the env config**
	"""
	def __init__(self, config, actor_rollout_wg, tokenizer):
		self.config = config
		self.train_ctx_manager = ContextManager(config, tokenizer, mode="train")
		self.train_es_manager = EnvStateManager(config, mode="train")
		self.val_ctx_manager = ContextManager(config, tokenizer, mode="val")
		self.val_es_manager = EnvStateManager(config, mode="val")
		self.actor_wg = actor_rollout_wg
		self.tokenizer = tokenizer

	def generate_sequences(self, lm_inputs: DataProto):
		# TODO: add kv cache both for the vllm wrapper here and for verl vllm.
		if isinstance(self.actor_wg, ApiCallingWrapperWg):
			lm_outputs = self.actor_wg.generate_sequences(lm_inputs)
		else:
			raise ValueError(f"Unsupported actor worker type: {type(self.actor_wg)}")

		return lm_outputs

	def rollout(self, dataproto: DataProto, val=False):
		es_manager = self.val_es_manager if val else self.train_es_manager
		ctx_manager = self.val_ctx_manager if val else self.train_ctx_manager
		env_outputs = es_manager.reset()

		for i in range(self.config.agent_proxy.max_turn):
			lm_inputs: DataProto = ctx_manager.get_lm_inputs(env_outputs, prepare_for_update=False)
			lm_inputs.meta_info = dataproto.meta_info # TODO: setup vllm early stop when max length is reached. make sure this can be done
			lm_outputs: DataProto = self.generate_sequences(lm_inputs)
			env_inputs: List[Dict] = ctx_manager.get_env_inputs(lm_outputs)
			env_outputs: List[Dict] = es_manager.step(env_inputs)
			if len(env_outputs) == 0: # all finished
				break
		rollout_states = es_manager.get_rollout_states()
		rollouts = ctx_manager.formulate_rollouts(rollout_states)
		# self.tokenizer.batch_decode(rollouts.batch['input_ids'], skip_special_tokens=False) # see all the trajectories
		return rollouts

# @hydra.main(version_base=None, config_path="../../config", config_name="base")
# def main(config):
# 	# detect config name from python -m ragen.llm_agent.agent_proxy --config_name frozen_lake
# 	tokenizer = AutoTokenizer.from_pretrained(config.actor_rollout_ref.model.path)
# 	actor_wg = ApiCallingWrapperWg(config, tokenizer)
# 	proxy = LLMAgentProxy(config, actor_wg, tokenizer)
# 	import time
# 	for _ in range(3):
# 		start_time = time.time()
# 		rollouts = proxy.rollout(DataProto(batch=None, non_tensor_batch=None, meta_info={'eos_token_id': 151645, 'pad_token_id': 151643, 'recompute_log_prob': False, 'do_sample':config.actor_rollout_ref.rollout.do_sample, 'validate': True}), val=True)
# 		end_time = time.time()
# 		print(f'rollout time: {end_time - start_time} seconds')
# 		# print rollout rewards from the rm_scores
# 		rm_scores = rollouts.batch["rm_scores"]
# 		metrics = rollouts.meta_info["metrics"]
# 		avg_reward = rm_scores.sum(-1).mean().item()
# 		print(f'rollout rewards: {avg_reward}')
# 		print(f'metrics:')
# 		for k, v in metrics.items():
# 			print(f'{k}: {v}')


@hydra.main(version_base=None, config_path="../../config", config_name="evaluate_api_llm")
def main(config):
	# detect config name from python -m ragen.llm_agent.agent_proxy --config_name frozen_lake
	tokenizer = AutoTokenizer.from_pretrained(config.actor_rollout_ref.model.path)
	actor_wg = ApiCallingWrapperWg(config, tokenizer)
	proxy = LLMAgentProxy(config, actor_wg, tokenizer)
	import time
	start_time = time.time()
	rollouts = proxy.rollout(DataProto(batch=None, non_tensor_batch=None, meta_info={'eos_token_id': 151645, 'pad_token_id': 151643, 'recompute_log_prob': False, 'do_sample': False, 'validate': True}), val=False)

	# print("messages_list: ", rollouts.non_tensor_batch['messages_list'])
	# print(f'[DEBUG] rollouts: {rollouts}')
	end_time = time.time()
	print(f'rollout time: {end_time - start_time} seconds')
	# print rollout rewards from the rm_scores
	rm_scores = rollouts.batch["rm_scores"].sum(-1)
	messages_list = rollouts.non_tensor_batch['messages_list']

	env_ids = rollouts.non_tensor_batch['env_ids']
	# print(f'[DEBUG] env_ids: {env_ids}')
	group_ids = rollouts.non_tensor_batch['group_ids']
	# print(f'[DEBUG] group_ids: {group_ids}')
	if not os.path.exists(config.output_dir):
		os.makedirs(config.output_dir)
	with open(f"{config.output_dir}/messages_list.txt", "w") as f:
		for i, msg in enumerate(messages_list):
			# print(i)
			# print("score: ", rm_scores[i])
			score = rm_scores[i].item()
			score = min(max(score, 0.0), 1.0)  # Ensure score is between -1 and 1
			row = {
				"request_id": f"environment_{group_ids[i]}",
				"messages": msg,
				"score": score,
			}
			f.write(json.dumps(row) + "\n")
	print(f'[DEBUG] rm_scores: {rm_scores.sum(-1)}')
	metrics = rollouts.meta_info["metrics"]
	avg_reward = rm_scores.sum(-1).mean().item()
	print(f'rollout rewards: {avg_reward}')
	print(f'metrics:')
	for k, v in metrics.items():
		print(f'{k}: {v}')



if __name__ == "__main__":
	main()

import time
import os
import socket
import arxiv
from loguru import logger
from llama_cpp import Llama
from abc import ABC, abstractmethod
from tldr import get_paper_summary
from utils import select_gpu_with_max_free_memory
from openai import OpenAI

class LLMBackend(ABC):
    @abstractmethod
    def get_paper_tldr(self, paper: arxiv.Result) -> str:
        raise NotImplementedError("get_paper_tldr is an abstract method and should be implemented in the subclass.")

    @staticmethod
    def get_backend(backend_type: str, model_path: str, **kwargs):
        if backend_type == "llama_cpp":
            return LlamaCppBackend(model_path)
        elif backend_type == "vllm":
            return VLLMBackend(model_path, **kwargs)
        else:
            raise ValueError(f"Invalid backend type: {backend_type}")
    
class LlamaCppBackend(LLMBackend):

    def __init__(self, model_path: str):
        self.model = Llama.from_pretrained(
            repo_id="Qwen/Qwen2.5-3B-Instruct-GGUF",
            filename="qwen2.5-3b-instruct-q4_k_m.gguf",
            n_ctx=4096,
            n_threads=4,
            verbose=False
        )

    def get_paper_tldr(self, paper: arxiv.Result) -> str:
        try:
            introduction, conclusion = get_paper_summary(paper)
        except:
            introduction, conclusion = "", ""
        prompt = """Given the title, abstract, introduction and the conclusion (if any) of a paper in latex format, generate a one-sentence TLDR summary:
        
        \\title{__TITLE__}
        \\begin{abstract}__ABSTRACT__\\end{abstract}
        __INTRODUCTION__
        __CONCLUSION__
        """
        prompt = prompt.replace('__TITLE__', paper.title)
        prompt = prompt.replace('__ABSTRACT__', paper.summary)
        prompt = prompt.replace('__INTRODUCTION__', introduction)
        prompt = prompt.replace('__CONCLUSION__', conclusion)
        prompt_tokens = self.model.tokenize(prompt.encode('utf-8'))
        prompt_tokens = prompt_tokens[:3800] # truncate to 3800 tokens
        prompt = self.model.detokenize(prompt_tokens).decode('utf-8')
        response = self.model.create_chat_completion(
            messages=[
              {"role": "system", "content": "You are an assistant who perfectly summarizes scientific paper, and gives the core idea of the paper to the user."},
              {
                  "role": "user",
              "content": prompt
            }
            ],
            top_k=0,
            temperature=0
        )
        return response['choices'][0]['message']['content']

class VLLMBackend(LLMBackend):

    def __init__(self, model_path: str, **kwargs):
        logger.info(f"Initializing VLLM backend with model path: {model_path}")
        self.model_path = model_path
        import subprocess

        cmd = ["vllm", "serve", model_path, "--dtype", "auto"]

        num_gpus = 1
        for key, value in kwargs.items():
            key = f"--{key.replace('_', '-')}"
            if key == "--tensor-parallel-size" or key == "--pipeline-parallel-size":
                cmd.extend([key, str(value)])
                num_gpus = num_gpus * value

        selected_gpus = select_gpu_with_max_free_memory(num_gpus)
        logger.info(f"vLLM Server Selected GPUs: {selected_gpus}")

        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, selected_gpus))
        
        self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # retry until the server is running
        time.sleep(20) # wait for loading weights

        port = kwargs.get("port", 8000)
        max_retries = 30 # wait for maximum 3 minutes
        while max_retries > 0:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.connect(('localhost', port))
                    logger.info(f"Server is running on port {port}")
                    break
            except:
                time.sleep(6)
                max_retries -= 1
                if max_retries == 0:
                    logger.error(f"Failed to start server on port {port}")
                    raise RuntimeError(f"Failed to start server on port {port}")


    def get_paper_tldr(self, paper: arxiv.Result) -> str:
        try:
            introduction, conclusion = get_paper_summary(paper)
        except:
            introduction, conclusion = "", ""
        prompt = """Given the title, abstract, introduction and the conclusion (if any) of a paper in latex format, generate a one-sentence TLDR summary:
        
        \\title{__TITLE__}
        \\begin{abstract}__ABSTRACT__\\end{abstract}
        __INTRODUCTION__
        __CONCLUSION__
        """
        prompt = prompt.replace('__TITLE__', paper.title)
        prompt = prompt.replace('__ABSTRACT__', paper.summary)
        prompt = prompt.replace('__INTRODUCTION__', introduction)
        prompt = prompt.replace('__CONCLUSION__', conclusion)

        openai_api_key = "EMPTY"
        openai_api_base = "http://localhost:8000/v1"

        client = OpenAI(
            # defaults to os.environ.get("OPENAI_API_KEY")
            api_key=openai_api_key,
            base_url=openai_api_base,
        )

        response = client.chat.completions.create(
            model=self.model_path,
            messages=[
              {"role": "system", "content": "You are an assistant who perfectly summarizes scientific paper, and gives the core idea of the paper to the user."},
              {
                  "role": "user",
                  "content": prompt
            }
            ],
            temperature=0
        )
        return response.choices[0].message.content

    def __del__(self):
        logger.info("Terminating VLLM server...")
        self.process.terminate()

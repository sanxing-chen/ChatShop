# ChatShop

[![Python version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/release/python-3813/)

Implementation of the ChatShop environment and search agents for the paper:

**[ChatShop: Interactive Information Seeking with Language Agents](https://arxiv.org/abs/2404.09911)**  
Sanxing Chen, Sam Wiseman, Bhuwan Dhingra

If you find this work useful in your research, please cite:

```
@misc{chen2024chatshop,
      title={ChatShop: Interactive Information Seeking with Language Agents}, 
      author={Sanxing Chen and Sam Wiseman and Bhuwan Dhingra},
      year={2024},
      eprint={2404.09911},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2404.09911}, 
}
```

## Setup

0. Clone the repository
1. Create a new conda environment with Python
    ``` shell
    > conda create -n chatshop python=3.8.13
    > conda activate chatshop
    ```
2. Run `setup_ws.sh -d all` to download product data from WebShop
3. Run `./run_server.sh` to start the ChatShop server
    - you can test it by running the following command:
        ``` shell
        curl "localhost:3000/api/search_results?keywords=protein%20shake&gid=1"
        ```
4. Test agents with the ChatShop environment
    ``` shell
    usage: ipofn.py [-h] [--model {gpt-3.5-turbo,gpt-3.5-turbo-1106,gpt-4-1106-preview}] [--strategy {random,allq,interleave}] [--cot] [--game_mode {full,part,subj}]
                    [--qmode {none,open,compare,mix}] [--num_question NUM_QUESTION]
                    exp_name

    positional arguments:
    exp_name              experiment name

    optional arguments:
    -h, --help            show this help message and exit
    --model {gpt-3.5-turbo,gpt-3.5-turbo-1106,gpt-4-1106-preview}
                            Agent: select chat model
    --strategy {random,allq,interleave}
                            Agent: action strategy
    --cot                 Agent: ReAct style prompting
    --game_mode {full,part,subj}
                            Env: task ambiguity mode, full is the original WebShop instructions, part is the instructions without salient attributes, subj is the
                            instructions without any attributes.
    --qmode {none,open,compare,mix}
                            Env: communication channel, none means no communication between agent and shopper, open enables open-ended communication, compare is the
                            instance-based communication.
    --num_question NUM_QUESTION
                            Env: maximum number of interactions with the simulated shopper.

    > python -m web_agent_site.models.ipofn test --game_mode subj --qmode open --num_question 1
    ```

## Acknowledgements

This repository sourced data and code from [WebShop](https://github.com/princeton-nlp/WebShop/).
import pandas as pd

ground_truth_manual = pd.DataFrame(
    [
        {
            'arxiv_id': '2608.06628',
            'citation_key': 'kim2025train',
            'ref_arxiv_id': '2502.06768'
        },
        {
            'arxiv_id': '2608.06628',
            'citation_key': 'ye2024beyond',
            'ref_arxiv_id': '2410.14157'
        },
        {
            'arxiv_id': '2608.06628',
            'citation_key': 'song2025seed',
            'ref_arxiv_id': '2508.02193'
        },
        {
            'arxiv_id': '2608.06628',
            'citation_key': 'trainin2026discrete',
            'ref_arxiv_id': '2602.19980'
        },
        {
            'arxiv_id': '2608.06628',
            'citation_key': 'schiff2025simple',
            'ref_arxiv_id': '2412.10193'
        },
        {
            'arxiv_id': '2608.06628',
            'citation_key': 'von2025scaling',
            'ref_arxiv_id': '2512.10858'
        },
        {
            'arxiv_id': '2608.06628',
            'citation_key': 'sahoo2025diffusion',
            'ref_arxiv_id': '2506.10892'
        },
        {
            'arxiv_id': '2608.06628',
            'citation_key': 'shi2024simplified',
            'ref_arxiv_id': '2406.04329'
        },
        {
            'arxiv_id': '2608.06628',
            'citation_key': 'sahoo2024simple',
            'ref_arxiv_id': '2406.07524'
        },
        {
            'arxiv_id': '2608.06628',
            'citation_key': 'gat2024discrete',
            'ref_arxiv_id': '2407.15595'
        },
        {
            'arxiv_id': '2608.06628',
            'citation_key': 'ye2025dream',
            'ref_arxiv_id': '2508.15487'
        },
        {
            'arxiv_id': '2608.06628',
            'citation_key': 'nie2025llada',
            'ref_arxiv_id': '2502.09992'
        },
        {
            'arxiv_id': '2608.06628',
            'citation_key': 'bie2026llada2',
            'ref_arxiv_id': '2602.08676'
        },
        {
            'arxiv_id': '2608.06628',
            'citation_key': 'labs2025mercury',
            'ref_arxiv_id': '2506.17298'
        },
        {
            'arxiv_id': '2608.06628',
            'citation_key': 'katharopoulos2020linear',
            'ref_arxiv_id': '2006.16236'
        },
        {
            'arxiv_id': '2608.06628',
            'citation_key': 'zhang2024hedgehog',
            'ref_arxiv_id': '2402.04347'
        },
        {
            'arxiv_id': '2608.06628',
            'citation_key': 'zhang2024lolcats',
            'ref_arxiv_id': '2410.10254'
        },
        {
            'arxiv_id': '2608.06628',
            'citation_key': 'lambert2024tulu3',
            'ref_arxiv_id': '2411.15124'
        },
        {
            'arxiv_id': '2608.06628',
            'citation_key': 'zheng2024sglang',
            'ref_arxiv_id': '2312.07104'
        },
        {
            'arxiv_id': '2608.06628',
            'citation_key': 'arriola2025block',
            'ref_arxiv_id': '2503.09573'
        },
        {
            'arxiv_id': '2608.06628',
            'citation_key': 'llada2scaling2025',
            'ref_arxiv_id': '2512.15745'
        },
        {
            'arxiv_id': '2608.06628',
            'citation_key': 'liu2025lawcat',
            'ref_arxiv_id': '2509.18467'
        },

        
        {
            'arxiv_id': '2608.06434',
            'citation_key': 'pi0',
            'ref_arxiv_id': '2410.24164'
        },
        {
            'arxiv_id': '2608.06434',
            'citation_key': 'openvla',
            'ref_arxiv_id': '2406.09246'
        },
        {
            'arxiv_id': '2608.06434',
            'citation_key': 'fast-in-slow',
            'ref_arxiv_id': '2506.01953'
        },
        {
            'arxiv_id': '2608.06434',
            'citation_key': 'humevla',
            'ref_arxiv_id': '2505.21432'
        },
        {
            'arxiv_id': '2608.06434',
            'citation_key': 'dualvla',
            'ref_arxiv_id': '2410.15549'
        },
        {
            'arxiv_id': '2608.06434',
            'citation_key': 'paligemma',
            'ref_arxiv_id': '2407.07726'
        },
        {
            'arxiv_id': '2608.06434',
            'citation_key': 'prismatic',
            'ref_arxiv_id': '2402.07865'
        },
        {
            'arxiv_id': '2608.06434',
            'citation_key': 'openX',
            'ref_arxiv_id': '2310.08864'
        },
        {
            'arxiv_id': '2608.06434',
            'citation_key': 'bridgedata',
            'ref_arxiv_id': '2308.12952'
        },
        {
            'arxiv_id': '2608.06434',
            'citation_key': 'tinyvla',
            'ref_arxiv_id': '2409.12514'
        },
        {
            'arxiv_id': '2608.06434',
            'citation_key': 'smolvla',
            'ref_arxiv_id': '2506.01844'
        },
        {
            'arxiv_id': '2608.06434',
            'citation_key': 'rt-2',
            'ref_arxiv_id': '2307.15818'
        },
        {
            'arxiv_id': '2608.06434',
            'citation_key': 'diffusionpolicy',
            'ref_arxiv_id': '2303.04137'
        },
        {
            'arxiv_id': '2608.06434',
            'citation_key': 'BCTransformer',
            'ref_arxiv_id': '2108.03298'
        },
        {
            'arxiv_id': '2608.06434',
            'citation_key': 'act',
            'ref_arxiv_id': '2304.13705'
        },
        {
            'arxiv_id': '2608.06434',
            'citation_key': 'clip',
            'ref_arxiv_id': '2103.00020'
        },
        {
            'arxiv_id': '2608.06434',
            'citation_key': 'r3m',
            'ref_arxiv_id': '2203.12601'
        },
        {
            'arxiv_id': '2608.06434',
            'citation_key': 'vc-1',
            'ref_arxiv_id': '2303.18240'
        },
        {
            'arxiv_id': '2608.06434',
            'citation_key': 'cliport',
            'ref_arxiv_id': '2109.12098'
        },
        {
            'arxiv_id': '2608.06434',
            'citation_key': 'gr00t',
            'ref_arxiv_id': '2503.14734'
        },
        {
            'arxiv_id': '2608.06434',
            'citation_key': 'hirt',
            'ref_arxiv_id': '2410.05273'
        },
        {
            'arxiv_id': '2608.06434',
            'citation_key': 'cogact',
            'ref_arxiv_id': '2411.19650'
        },
        {
            'arxiv_id': '2608.06434',
            'citation_key': 'robodual',
            'ref_arxiv_id': '2410.08001'
        },
        {
            'arxiv_id': '2608.06434',
            'citation_key': 'trivla',
            'ref_arxiv_id': '2507.01424'
        },
        {
            'arxiv_id': '2608.06434',
            'citation_key': 'iql',
            'ref_arxiv_id': '2110.06169'
        },
        {
            'arxiv_id': '2608.06434',
            'citation_key': 'libero',
            'ref_arxiv_id': '2306.03310'
        },
        {
            'arxiv_id': '2608.06434',
            'citation_key': 'dqn',
            'ref_arxiv_id': '1710.02298'
        },


        {
            'arxiv_id': '2608.04791',
            'citation_key': 'guo2023promptfl',
            'ref_arxiv_id': '2208.11625'
        },
        {
            'arxiv_id': '2608.04791',
            'citation_key': 'hou2025capt',
            'ref_arxiv_id': '2503.06993'
        },
        {
            'arxiv_id': '2608.04791',
            'citation_key': 'pan2024federated',
            'ref_arxiv_id': '2409.19610'
        },
        {
            'arxiv_id': '2608.04791',
            'citation_key': 'li2024global',
            'ref_arxiv_id': '2403.00041'
        },
        {
            'arxiv_id': '2608.04791',
            'citation_key': 'cui2024harmonizing',
            'ref_arxiv_id': '2405.09771'
        },
        {
            'arxiv_id': '2608.04791',
            'citation_key': 'lu2023fedclip',
            'ref_arxiv_id': '2302.13485'
        },
        {
            'arxiv_id': '2608.04791',
            'citation_key': 'wu2025faa',
            'ref_arxiv_id': '2503.05776'
        },
        {
            'arxiv_id': '2608.04791',
            'citation_key': 'mitra2025fedvlm',
            'ref_arxiv_id': '2507.17088'
        },
        {
            'arxiv_id': '2608.04791',
            'citation_key': 'clasen_reben_2025',
            'ref_arxiv_id': '2407.03653'
        },
        {
            'arxiv_id': '2608.04791',
            'citation_key': 'goyal_finetune_2023',
            'ref_arxiv_id': '2212.00638'
        },
        {
            'arxiv_id': '2608.04791',
            'citation_key': 'hu_lora_2022',
            'ref_arxiv_id': '2106.09685'
        },
        {
            'arxiv_id': '2608.04791',
            'citation_key': 'ilharco_patching_2022',
            'ref_arxiv_id': '2208.05592'
        },
        {
            'arxiv_id': '2608.04791',
            'citation_key': 'zhou_learning_2022',
            'ref_arxiv_id': '2109.01134'
        },
        {
            'arxiv_id': '2608.04791',
            'citation_key': 'russakovsky_imagenet_2014',
            'ref_arxiv_id': '1409.0575'
        },
        {
            'arxiv_id': '2608.04791',
            'citation_key': 'cheng_remote_2017',
            'ref_arxiv_id': '1703.00121'
        },
        {
            'arxiv_id': '2608.04791',
            'citation_key': 'helber_eurosat_2019',
            'ref_arxiv_id': '1709.00029'
        },
        {
            'arxiv_id': '2608.04791',
            'citation_key': 'buyuktacs2024federated',
            'ref_arxiv_id': '2311.06141'
        },
        {
            'arxiv_id': '2608.04791',
            'citation_key': 'xie2023fedkl',
            'ref_arxiv_id': '2204.08125'
        },
        {
            'arxiv_id': '2608.04791',
            'citation_key': 'lin2025fedrsclip',
            'ref_arxiv_id': '2501.02461'
        },
        {
            'arxiv_id': '2608.04791',
            'citation_key': 'jiminez2024non',
            'ref_arxiv_id': '2411.12377'
        },
        {
            'arxiv_id': '2608.04791',
            'citation_key': 'mcmahan2017communication',
            'ref_arxiv_id': '1602.05629'
        },
        {
            'arxiv_id': '2608.04791',
            'citation_key': 'radford2021learning',
            'ref_arxiv_id': '2103.00020'
        },


        {
            'arxiv_id': '2608.06791',
            'citation_key': 'cong2018besteffort',
            'ref_arxiv_id': '1807.01340'
        },
        {
            'arxiv_id': '2608.06791',
            'citation_key': 'abikaram2024hlsfactory',
            'ref_arxiv_id': '2405.00820'
        },
        {
            'arxiv_id': '2608.06791',
            'citation_key': 'abikaram2025hlseval',
            'ref_arxiv_id': '2504.12268'
        },
        {
            'arxiv_id': '2608.06791',
            'citation_key': 'gai2025hlsmodels',
            'ref_arxiv_id': '2502.13921'
        },
        {
            'arxiv_id': '2608.06791',
            'citation_key': 'xiong2024hlspilot',
            'ref_arxiv_id': '2408.06810'
        },
        {
            'arxiv_id': '2608.06791',
            'citation_key': 'li2026chathls',
            'ref_arxiv_id': '2507.00642'
        },
        {
            'arxiv_id': '2608.06791',
            'citation_key': 'mashnoor2025timelyhls',
            'ref_arxiv_id': '2507.17962'
        },
        {
            'arxiv_id': '2608.06791',
            'citation_key': 'prakriya2025lift',
            'ref_arxiv_id': '2504.21187'
        },
        {
            'arxiv_id': '2608.06791',
            'citation_key': 'moraru2026laafd',
            'ref_arxiv_id': '2602.06085'
        },
        {
            'arxiv_id': '2608.06791',
            'citation_key': 'bhandwaldar2026agentfactories',
            'ref_arxiv_id': '2603.25719'
        },
        {
            'arxiv_id': '2608.06791',
            'citation_key': 'zou2026agrefactor',
            'ref_arxiv_id': '2606.30949'
        },
        {
            'arxiv_id': '2608.06791',
            'citation_key': 'li2026cudahercules',
            'ref_arxiv_id': '2605.08467'
        },
        {
            'arxiv_id': '2608.06791',
            'citation_key': 'xu2026deepseek',
            'ref_arxiv_id': '2606.19348'
        },
        {
            'arxiv_id': '2608.06791',
            'citation_key': 'zou2026hlsseek',
            'ref_arxiv_id': '2605.13536'
        },
        {
            'arxiv_id': '2608.06791',
            'citation_key': 'sohrabizadeh2020autodse',
            'ref_arxiv_id': '2009.14381'
        },
        {
            'arxiv_id': '2608.06791',
            'citation_key': 'ye2021scalehls',
            'ref_arxiv_id': '2107.11673'
        },
        {
            'arxiv_id': '2608.06791',
            'citation_key': 'pouget2025sisyphus',
            'ref_arxiv_id': '2405.03058'
        },
        {
            'arxiv_id': '2608.06791',
            'citation_key': 'dolan2002performance',
            'ref_arxiv_id': 'cs/0102001'
        },
        {
            'arxiv_id': '2608.06791',
            'citation_key': 'Xu2024OptimizingModels',
            'ref_arxiv_id': '2410.07356'
        },
        {
            'arxiv_id': '2608.06791',
            'citation_key': 'collini2024c2hlsc',
            'ref_arxiv_id': '2412.00214'
        },
        {
            'arxiv_id': '2608.06791',
            'citation_key': 'dettmers2023qlora',
            'ref_arxiv_id': '2305.14314'
        },
        {
            'arxiv_id': '2608.06791',
            'citation_key': 'shao2024deepseekmath',
            'ref_arxiv_id': '2402.03300'
        },


        {
            'arxiv_id': '2608.05850',
            'citation_key': 'sentalign-2023',
            'ref_arxiv_id': '2311.08982'
        },
        {
            'arxiv_id': '2608.05850',
            'citation_key': 'wu2025bitterlessonlearned2000',
            'ref_arxiv_id': '2504.15521'
        },
        {
            'arxiv_id': '2608.05850',
            'citation_key': 'urieli2025jochre',
            'ref_arxiv_id': '2501.08442'
        },
        {
            'arxiv_id': '2608.05850',
            'citation_key': 'koto2025sherkalachat',
            'ref_arxiv_id': '2503.01493'
        },
        {
            'arxiv_id': '2608.05850',
            'citation_key': 'kulick2022part',
            'ref_arxiv_id': '2204.01175'
        },
        {
            'arxiv_id': '2608.05850',
            'citation_key': 'grattafiori2024llama',
            'ref_arxiv_id': '2407.21783'
        },
        {
            'arxiv_id': '2608.05850',
            'citation_key': 'yang2025qwen3',
            'ref_arxiv_id': '2505.09388'
        },
        {
            'arxiv_id': '2608.05850',
            'citation_key': 'bigscience-roots:2022',
            'ref_arxiv_id': '2303.03915'
        },
        {
            'arxiv_id': '2608.05850',
            'citation_key': 'usco2025ai',
            'ref_arxiv_id': '2303.03915'
        },
        
        
        
    ]
)
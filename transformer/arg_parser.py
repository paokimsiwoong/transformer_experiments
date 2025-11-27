import os

from argparse import ArgumentParser

import torch

def parse_args():
    parser = ArgumentParser()

    # Conventional args

    # 학습 데이터 경로
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.environ.get(
            "SM_DATA_DIR",
            "/home/paokimsiwoong/workspace/github.com/paokimsiwoong/transformer_experiments/transformer/data.csv",
        ),
    )

    # pth 파일 저장 경로
    parser.add_argument(
        "--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/home/paokimsiwoong/workspace/github.com/paokimsiwoong/transformer_experiments/transformer/pths")
    )

    # png 파일 저장 경로
    parser.add_argument(
        "--image_dir", type=str, default=os.environ.get("SM_IMAGE_DIR", "/home/paokimsiwoong/workspace/github.com/paokimsiwoong/transformer_experiments/transformer/attention_viz")
    )
   

    # resume 파일 이름
    parser.add_argument("--resume_name", type=str, default="")

    # random seed
    parser.add_argument("--seed", type=int, default=42)


    # attention visualization 여부
    parser.add_argument('--viz', action='store_true', default=False, help='enable visualization mode')
    # debug 여부
    parser.add_argument('--debug', action='store_true', default=False, help='enable debug mode')

    # 루프 조기종료 여부
    parser.add_argument('--train_break', action='store_true', default=False, help='break train loop at step 2')
    parser.add_argument('--val_break', action='store_true', default=False, help='break val loop at step 2')
    parser.add_argument('--test_break', action='store_true', default=False, help='break test loop at step 2')

    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--num_workers", type=int, default=4)
    # gpu 개수 * 4로 설정하는 것이 좋다는 의견이 있음

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--val_batch_size", type=int, default=32)
    parser.add_argument("--test_batch_size", type=int, default=32)
    parser.add_argument("--val_num_workers", type=int, default=4)

    parser.add_argument("--max_token_length", type=int, default=512)
    parser.add_argument("--q_dim", type=int, default=512)

    parser.add_argument('--weight_tying', action='store_false', default=True, help='enable weight tying')
    parser.add_argument('--decouple_src_tgt_embed', action='store_true', default=False, help='decouple src tgt embed weights')

    parser.add_argument("--len_vocab", type=int, default=64101)
    parser.add_argument("--start_idx", type=int, default=64100)
    parser.add_argument("--end_idx", type=int, default=1)
    parser.add_argument("--padding_idx", type=int, default=0)
    parser.add_argument("--unk_idx", type=int, default=2)


    parser.add_argument("--label_smoothing", type=float, default=0.1)


    parser.add_argument("--learning_rate", type=float, default=1)
    parser.add_argument("--warmup_steps", type=int, default=16000)
    # @@@ 전체 step의 5~10% (128만 문장을 batch_size 8 => 160000 스텝 => 10%는 16000)
    # parser.add_argument("--warmup_steps", type=int, default=4000)
    # @@@ 논문은 100000 step의 4% 설정 
    # @@@ @@@ 450만 문장을 각 step당 2만5천씩 한 epoch에 180 step 정도 진행
    # @@@ @@@ 대략 556 epoch (100000/180) 진행
    # parser.add_argument("--warmup_steps", type=int, default=30)
    # @@@ 논문 장비 기준으로 현재 데이터셋은 한 epoch에 50 step
    # @@@ ==> 3 epoch 학습 시 warmup을 15, 6 epoch 학습 시 30 (10% 기준)

    # # rate 함수 변경
    # parser.add_argument("--learning_rate", type=float, default=0.0005)
    # parser.add_argument("--warmup_steps", type=int, default=50000)
    # parser.add_argument("--total_steps", type=int, default=1000000)
    # parser.add_argument("--decay_rate", type=float, default=0.7)

    # gradient clipping 값
    parser.add_argument("--max_norm", type=float, default=1.0)

    parser.add_argument("--max_epoch", type=int, default=10)

    parser.add_argument("--save_interval", type=int, default=1)
    parser.add_argument("--val_interval", type=int, default=1)

    # ????

    parser.add_argument('--mp', action='store_true', default=False, help='enable mixed precision mode')
    # https://stackoverflow.com/questions/60999816/argparse-not-parsing-boolean-arguments
    # mixed precision 사용할 지 여부

    # parser.add_argument("--wandb_mode", type=str, default="online")
    # parser.add_argument("--wandb_mode", type=str, default="offline")
    parser.add_argument("--wandb_mode", type=str, default="disabled")
    # wandb mode
    parser.add_argument("--wandb_run_name", type=str, default="KorEnTransformer")
    # wandb run name

    args = parser.parse_args()

    return args

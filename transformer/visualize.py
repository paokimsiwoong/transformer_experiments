import os
import os.path as osp

import pandas as pd

import altair as alt

import warnings

# UserWarning 중에서 메시지에 'Automatically deduplicated selection parameter'가 포함된 경고 무시
warnings.filterwarnings("ignore", message="Automatically deduplicated selection parameter", category=UserWarning)

def visualize(image_dir, log_name, step, model, loaders, cat_list, inputs, preds, labels, n_examples=4):
    if not osp.exists(image_dir):
        os.makedirs(image_dir)

    folder_path = osp.join(image_dir, log_name)

    if not osp.exists(folder_path):
        os.makedirs(folder_path)

    sub_folder_path = osp.join(folder_path, f'{step}')

    if not osp.exists(sub_folder_path):
        os.makedirs(sub_folder_path)

    loop_len = min(inputs.size(0), n_examples)

    results = [()] * loop_len

    for i in range(loop_len):
        # @@@@ 한 문장의 토큰들을 다 합친 결과가 아니라 토큰들이 각 원소로 들어가있어야 함
        decoded_inputs = loaders.tokenizer.convert_ids_to_tokens(inputs[i])

        decoded_preds = loaders.tokenizer.convert_ids_to_tokens(preds[i])

        decoded_labels = loaders.tokenizer.convert_ids_to_tokens(labels[i])

        # Some simple post-processing
        decoded_inputs = [inp.strip() for inp in decoded_inputs if inp != "<pad>"]
        decoded_preds = [pred.strip() for pred in decoded_preds if pred != "<pad>"]
        decoded_labels = [label.strip() for label in decoded_labels if label != "<pad>"]
        # batch size가 1이면 loader의 collate_fn 설정에 의해 pad 토큰이 없지만 일단 유지

        print(
            "Source Text (Input)        : "
            + " ".join(decoded_inputs)
        )
        print(
            "Target Text (Ground Truth) : "
            + " ".join(decoded_labels)
        )
        print("Model Output             : " 
              + " ".join(decoded_preds)
        )
        results[i] = (cat_list[i], decoded_inputs, decoded_labels, preds[i], decoded_preds)


    for j, r in enumerate(results):
        file_name = osp.join(sub_folder_path, f"encoder_self_{j}.html")
        viz_encoder_self(model, j, r, file_name)
        file_name = osp.join(sub_folder_path, f"decoder_self_{j}.html")
        viz_decoder_self(model, j, r, file_name)
        file_name = osp.join(sub_folder_path, f"decoder_src_{j}.html")
        viz_decoder_src(model, j, r, file_name)




def mtx2df(m, max_row, max_col, row_tokens, col_tokens):
    "convert a dense matrix to a data frame with row and column indices"
    # m은 (q_seq_len, k_seq_len) 형태
    return pd.DataFrame(
        [
            (
                r,
                c,
                float(m[r, c]),
                "%.3d %s"
                % (r, row_tokens[r] if len(row_tokens) > r else "<blank>"),
                "%.3d %s"
                % (c, col_tokens[c] if len(col_tokens) > c else "<blank>"),
            )
            for r in range(m.shape[0])
            for c in range(m.shape[1])
            if r < max_row and c < max_col
        ],
        # if float(m[r,c]) != 0 and r < max_row and c < max_col],
        columns=["row", "column", "value", "row_token", "col_token"],
    )
    # [(row, column, attention 값, "row row토큰", "column column토큰"), ....] 형태의 리스트를
    # columns=["row", "column", "value", "row_token", "col_token"]로 열 이름을 설정한 뒤 pd DF로 생성
    # # @@@ query가 row, key가 col ==> decoder의 cross attention에서는 query가 영어, key가 한국어

def attn_map(attn, batch, head, row_tokens, col_tokens, max_dim=30):
    df = mtx2df(
        attn[batch, head].data, # attn은 (batch, head, q_seq_len, k_seq_len) ==> 지정된 head의 (q_seq_len, k_seq_len) 부분만 추출
        max_dim,
        max_dim,
        row_tokens,
        col_tokens,
    )

    # param = alt.selection_point(name=f'head{head}')

    return (
        alt.Chart(data=df)
        .mark_rect()
        .encode(
            x=alt.X("col_token", axis=alt.Axis(title="")), # 그래프 X축 == col(key)
            y=alt.Y("row_token", axis=alt.Axis(title="")), # 그래프 Y축 == row(query)
            color="value",
            tooltip=["row", "column", "value", "row_token", "col_token"],
        )
        .properties(height=400, width=400)
        .interactive()
        # .add_params(param)
    )


def get_encoder(model, layer):
    return model.encoder.blocks[layer].MHA.attn


def get_decoder_self(model, layer):
    return model.decoder.blocks[layer].MMHA.attn


def get_decoder_src(model, layer):
    return model.decoder.blocks[layer].MHA.attn

def visualize_layer(model, layer, getter_fn, batch, ntokens, row_tokens, col_tokens):
    # ntokens = last_example[0].ntokens
    attn = getter_fn(model, layer)
    n_heads = attn.shape[1]

    charts = [
        attn_map(
            attn,
            batch,
            h,
            row_tokens=row_tokens,
            col_tokens=col_tokens,
            max_dim=ntokens,
        )
        for h in range(n_heads)
    ]
    assert n_heads == 8

    combined_chart = alt.vconcat(
        charts[0]
        # | charts[1]
        | charts[2]
        # | charts[3]
        | charts[4]
        # | charts[5]
        | charts[6]
        # | charts[7]
        # layer + 1 due to 0-indexing
    ).properties(title="Layer %d" % (layer + 1))


    return combined_chart

def viz_encoder_self(model, batch, r, file_name):
    layer_viz = [
        visualize_layer(
            model, layer, get_encoder, batch, len(r[1]), r[1], r[1] # r[1]은 decoded_inputs ==> len(r[1])로 input의 토큰 개수 입력
        )
        for layer in range(6)
    ]

    encoder_self_chart = alt.hconcat(
        layer_viz[0]
        # & layer_viz[1]
        & layer_viz[2]
        # & layer_viz[3]
        & layer_viz[4]
        # & layer_viz[5]
    )


    encoder_self_chart.save(file_name)

def viz_decoder_self(model, batch, r, file_name):
    layer_viz = [
        visualize_layer(
            model,
            layer,
            get_decoder_self,
            batch,
            len(r[4]),
            r[4],
            r[4],
            # r[4]은 decoded_preds
        )
        for layer in range(6)
    ]

    decoder_self_chart = alt.hconcat(
        layer_viz[0]
        & layer_viz[1]
        & layer_viz[2]
        & layer_viz[3]
        & layer_viz[4]
        & layer_viz[5]
    )

    decoder_self_chart.save(file_name)

def viz_decoder_src(model, batch, r, file_name):
    layer_viz = [
        visualize_layer(
            model,
            layer,
            get_decoder_src,
            batch,
            max(len(r[1]),len(r[4])),
            r[4], # @@@ decoder의 cross attention query는 ys(preds) 
            r[1],
            # r[4]은 decoded_preds
            # r[1]은 decoded_inputs
        )
        for layer in range(6)
    ]

    decoder_src_chart = alt.hconcat(
        layer_viz[0]
        & layer_viz[1]
        & layer_viz[2]
        & layer_viz[3]
        & layer_viz[4]
        & layer_viz[5]
    )

    decoder_src_chart.save(file_name)


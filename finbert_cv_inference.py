import sys

import cv_inference

def run(
        mda_manual_folder_path: str,
        mda_tone_df_path: str,
        test_fold_index: int,
        result_folder_path: str,
        *,
        model_folder_path: str | None = None,
        gradient_accumulation_steps: int = 3
):
    cv_inference.run(
        mda_manual_folder_path=mda_manual_folder_path,
        mda_tone_df_path=mda_tone_df_path,
        test_fold_index=test_fold_index,
        result_folder_path=result_folder_path,
        model_folder_path=model_folder_path,
        gradient_accumulation_steps=gradient_accumulation_steps,
        base_model_path="ProsusAI/finbert"
    )

if __name__ == "__main__":
    mda_manual_folder_path = sys.argv[1]
    mda_tone_df_path = sys.argv[2]
    test_fold_index_str = sys.argv[3]
    result_folder_path = sys.argv[4]

    if len(result_folder_path) >= 6:
        model_folder_path = sys.argv[5]
    else:
        model_folder_path = None

    test_fold_index = int(test_fold_index_str)
    
    run(
        mda_manual_folder_path,
        mda_tone_df_path,
        test_fold_index,
        result_folder_path,
        model_folder_path=model_folder_path
    )

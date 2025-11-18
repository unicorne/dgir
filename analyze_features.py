import sys
path_to_pip_installs = "/tmp/test_env"
if path_to_pip_installs not in sys.path:
    sys.path.insert(0, path_to_pip_installs)

import torch
import numpy as np
import pandas as pd
from guided_diffusion.script_util import create_model_and_diffusion
from guided_diffusion.nn import timestep_embedding
from src.diffusion_registration.core.losses import NewLNCC
from monai.metrics import DiceMetric
dice_metric = DiceMetric(include_background=True, reduction="mean")
import monai
LNCC_loss = monai.losses.LocalNormalizedCrossCorrelationLoss(spatial_dims=2)

device = "cuda" if torch.cuda.is_available() else "cpu"

def transform_img(img):
    # transform to torch tensor
    img = torch.from_numpy(img).float()
    img = img.unsqueeze(0).unsqueeze(0)
    return img

def get_data_split(split):
    data_dixon = np.load(f"/home/students/studweilc1/MU-Diff/data/my_data3/{split}/DIXON.npy")
    data_t1 = np.load(f"/home/students/studweilc1/MU-Diff/data/my_data3/{split}/T1_mapping_fl2d.npy")
    data_dixon_masks = np.load(f"/home/students/studweilc1/MU-Diff/data/my_data3/{split}/DIXON_masks.npy")
    data_t1_masks = np.load(f"/home/students/studweilc1/MU-Diff/data/my_data3/{split}/T1_mapping_fl2d_masks.npy")
    data_dixon = transform_img(data_dixon)
    data_t1 = transform_img(data_t1)
    data_dixon_masks = transform_img(data_dixon_masks)
    data_t1_masks = transform_img(data_t1_masks)
    return data_t1, data_dixon, data_t1_masks, data_dixon_masks

def get_data(splits=["test"]):
    data_t1, data_dixon, data_t1_masks, data_dixon_masks = None, None, None, None
    for split in splits:
        data_t1, data_dixon, data_t1_masks, data_dixon_masks = get_data_split(split)

        data_t1 = data_t1 if data_t1 is None else torch.cat((data_t1, data_t1), dim=2)
        data_dixon = data_dixon if data_dixon is None else torch.cat((data_dixon, data_dixon), dim=2)
        data_t1_masks = data_t1_masks if data_t1_masks is None else torch.cat((data_t1_masks, data_t1_masks), dim=2)    
        data_dixon_masks = data_dixon_masks if data_dixon_masks is None else torch.cat((data_dixon_masks, data_dixon_masks), dim=2)

    return data_t1, data_dixon, data_t1_masks, data_dixon_masks

def define_loss_functions(diffusion, model_org, model_finetuned):
    new_lncc_org = NewLNCC(
        diffusion=diffusion,
        model=model_org,
        sigma=1.0,
        up_ft_index=10,
        t=60,
        use_lncc=True
    )

    new_lncc_finetuned = NewLNCC(
        diffusion=diffusion,
        model=model_finetuned,
        sigma=1.0,
        up_ft_index=10,
        t=60,
        use_lncc=True
    )

    new_lncc_org_mse = NewLNCC(
        diffusion=diffusion,
        model=model_org,
        sigma=1.0,
        up_ft_index=10,
        t=60,
        use_lncc=False
    )

    new_lncc_finetuned_mse = NewLNCC(
        diffusion=diffusion,
        model=model_finetuned,
        sigma=1.0,
        up_ft_index=10,
        t=60,
        use_lncc=False
    )


    new_lncc_org_6 = NewLNCC(
        diffusion=diffusion,
        model=model_org,
        sigma=1.0,
        up_ft_index=6,
        t=60,
        use_lncc=True
    )

    new_lncc_finetuned_6 = NewLNCC(
        diffusion=diffusion,
        model=model_finetuned,
        sigma=1.0,
        up_ft_index=6,
        t=60,
        use_lncc=True
    )

    new_lncc_finetuned_4 = NewLNCC(
        diffusion=diffusion,
        model=model_finetuned,
        sigma=1.0,
        up_ft_index=4,
        t=60,
        use_lncc=True
    )

    new_lncc_finetuned_8 = NewLNCC(
        diffusion=diffusion,
        model=model_finetuned,
        sigma=1.0,
        up_ft_index=8,
        t=60,
        use_lncc=True
    )

    new_lncc_finetuned_12 = NewLNCC(
        diffusion=diffusion,
        model=model_finetuned,
        sigma=1.0,
        up_ft_index=8,
        t=60,
        use_lncc=True
    )

    new_lncc_finetuned_14 = NewLNCC(
        diffusion=diffusion,
        model=model_finetuned,
        sigma=1.0,
        up_ft_index=8,
        t=60,
        use_lncc=True
    )

    new_lncc_finetuned_14_mse = NewLNCC(
        diffusion=diffusion,
        model=model_finetuned,
        sigma=1.0,
        up_ft_index=8,
        t=60,
        use_lncc=False
    )

    new_lncc_org_6_mse = NewLNCC(
        diffusion=diffusion,
        model=model_org,
        sigma=1.0,
        up_ft_index=6,
        t=60,
        use_lncc=False
    )

    new_lncc_finetuned_6_mse = NewLNCC(
        diffusion=diffusion,
        model=model_finetuned,
        sigma=1.0,
        up_ft_index=6,
        t=60,
        use_lncc=False
    )

    new_lncc_finetuned_2 = NewLNCC(
        diffusion=diffusion,
        model=model_finetuned,
        sigma=1.0,
        up_ft_index=2,
        t=60,
        use_lncc=True
    )

    new_lncc_finetuned_2_mse = NewLNCC(
        diffusion=diffusion,
        model=model_finetuned,
        sigma=1.0,
        up_ft_index=2,
        t=60,
        use_lncc=True
    )

    loss_dict = {
        "new_lncc_org": new_lncc_org,
        "new_lncc_finetuned": new_lncc_finetuned,
        "new_lncc_org_mse": new_lncc_org_mse,
        "new_lncc_finetuned_mse": new_lncc_finetuned_mse,
        "new_lncc_org_6": new_lncc_org_6,
        "new_lncc_finetuned_6": new_lncc_finetuned_6,
        "new_lncc_org_6_mse": new_lncc_org_6_mse,
        "new_lncc_finetuned_6_mse": new_lncc_finetuned_6_mse,
        "new_lncc_finetuned_4": new_lncc_finetuned_4,
        "new_lncc_finetuned_8": new_lncc_finetuned_8,
        "new_lncc_finetuned_12": new_lncc_finetuned_12,
        "new_lncc_finetuned_14": new_lncc_finetuned_14,
        "new_lncc_finetuned_2": new_lncc_finetuned_2,
        "new_lncc_finetuned_2_mse": new_lncc_finetuned_2_mse
    }
    return loss_dict

def define_models():
    diffusion_config = dict(
        image_size=256,
        class_cond=False, #
        learn_sigma=True, #
        num_channels=256, #
        num_res_blocks=2, #
        channel_mult="", # Will be auto-set
        num_heads=4, #
        num_head_channels=64, #
        num_heads_upsample=-1, #
        attention_resolutions="32,16,8", #
        dropout=0.0, #
        diffusion_steps=1000, #
        noise_schedule="linear", #
        timestep_respacing="", #
        use_kl=False, #
        predict_xstart=False, #
        rescale_timesteps=False, #
        rescale_learned_sigmas=False, #
        use_checkpoint=False, #
        use_scale_shift_norm=True, #
        resblock_updown=True, #
        use_fp16=False, #
        use_new_attention_order=False, #
    )

    model_path = "/home/students/studweilc1/MU-Diff/data/models/guided_diffusion/model180000.pt"
    print(f"Loading diffusion model onto {device}...")
    model_finetuned, diffusion = create_model_and_diffusion(**diffusion_config)
    model_finetuned.load_state_dict(torch.load(model_path, map_location="cpu"))
    model_finetuned.to(device)
    model_finetuned.eval()
    print("Model loaded.")

    model_path_org = "/home/students/studweilc1/dgir/guided_diffusion/256x256_diffusion_uncond.pt"
    print(f"Loading diffusion model onto {device}...")
    model_org, diffusion = create_model_and_diffusion(**diffusion_config)
    model_org.load_state_dict(torch.load(model_path_org, map_location="cpu"))
    model_org.to(device)
    model_org.eval()
    print("Model loaded.")
    return diffusion, model_org, model_finetuned

def main():
    diffusion, model_org, model_finetuned = define_models()
    data_t1, data_dixon, data_t1_masks, data_dixon_masks = get_data(splits=["train", "val", "test"])
    data_t1 = data_t1.to(device)
    data_dixon = data_dixon.to(device)
    data_t1_masks = data_t1_masks.to(device)
    data_dixon_masks = data_dixon_masks.to(device)

    print("Shape of T1 data:", data_t1.shape)

    loss_dict = define_loss_functions(diffusion, model_org, model_finetuned)

    all_columns = ["index", "dice_score"] + list(loss_dict.keys())
    df = pd.DataFrame(columns=all_columns)

    for index in range(data_t1.shape[2]):
        img_t1 = data_t1[:,:,index,:,:]
        img_dixon = data_dixon[:,:,index,:,:]
        mask_t1 = data_t1_masks[:,:,index,:,:]
        mask_dixon = data_dixon_masks[:,:,index,:,:]

        tmp_dce = dice_metric(mask_t1.to(device), mask_dixon.to(device)).item()
        tmp_lncc = LNCC_loss(img_t1, img_dixon).item()
        row = {"index": index}
        with torch.no_grad():
            for loss_name, loss_fn in loss_dict.items():
                loss_value = loss_fn(img_t1, img_dixon).item()
                row[loss_name] = loss_value

        row["dice_score"] = tmp_dce
        row["lncc_score"] = tmp_lncc

        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        print(f"Processed index {index}")

    df.to_csv("feature_analysis_results.csv", index=False)
    print("Results saved to feature_analysis_results.csv")

if __name__ == "__main__":
    main()


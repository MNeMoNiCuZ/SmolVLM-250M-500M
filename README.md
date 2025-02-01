# SmolVLM-500M-Instruct - Batch
This tool utilizes the SmolVLM-500M-Instruct model to automatically caption multiple image files in an input-folder, for ease of process.

Place all images you wish to caption in the /input directory and run `py batch.py`.

# Setup
1. Git clone this repository `git clone https://github.com/MNeMoNiCuZ/SmolVLM-500M-Instruct`
2. (Optional) Create a virtual environment for your setup. Use python 3.10 to 3.11. Feel free to use the `venv_create.bat` for a simple windows setup. Activate your venv.
3. Run `pip install -r requirements.txt` (this is done automatically with the `venv_create.bat`).
4. Install [PyTorch with CUDA support](https://pytorch.org/) matching your installed CUDA version. Run `nvcc --version` to find out which CUDA is your default.

You should now be set up and able to run the tool.

# Requirements
- Tested on Python 3.10.
- Tested on Pytorch w. CUDA 12.1.

- At batch size 1 it uses ~4gb VRAM
- At batch size 10 it uses ~23.6gb VRAM

Captioning takes around 20 seconds per batch.

# Examples
![creature](https://github.com/user-attachments/assets/4d2f8b51-52d5-4036-9ea1-8cab21ae21eb)

Input prompt:

`Describe this image in detail.`

```
The image depicts a small, furry creature with large, outstretched wings. The creature appears to be a squirrel, characterized by its bushy tail and small, round body. The fur on its head is dark, possibly black or dark brown, and it has a lighter-colored underbelly. The wings are a mix of dark and light shades, with the darker feathers forming the outer edges and the lighter feathers forming the inner edges. The wings are also quite large, almost as large as the creature itself.

The creature is sitting on a rocky surface, which appears to be a dirt or gravel area. The ground is uneven, with small rocks and pebbles scattered around. In the background, there is a large tree trunk, which is partially obscured by the creature's wings. The tree has a rough texture, with visible bark and branches. The tree's trunk is dark, contrasting with the lighter-colored wings.

The lighting in the image is somewhat diffuse, with no strong shadows. The background is dark, likely indicating a forest or a cave environment. The overall atmosphere of the image is serene and natural, with the creature being the focal point.

### Analysis and Description:

The creature's wings are a striking feature, and they suggest that it might be a type of squirrel or a similar small mammal. The fur on its head and underbelly is dark, which could be a reference to the natural color of the creature's fur. The wings are a mix of dark and light shades, which could indicate that the creature is a nocturnal or semi-aquatic animal, as their fur is not typically dark.

The tree in the background adds a natural element to the scene, and the overall composition suggests a natural, undisturbed environment. The lighting in the image is soft, with no harsh shadows, which helps to create a calm and serene atmosphere.

### Answer to Potential Questions:

**Question: What is the creature's fur color?**

**Answer:** The creature's fur is dark, possibly black or dark brown.

**Question: What is the background of the image?**

**Answer:** The background of the image is a large tree trunk, which is partially obscured by the creature's wings.

**Question: What is the creature's posture?**

**Answer:** The creature is sitting on a rocky surface, which suggests it might be resting or observing its surroundings.

**
```
---
Input prompt:

`describe`

> A squirrel with large, outstretched, grey, feathered ears sits on a dirt ground in front of a tree trunk.

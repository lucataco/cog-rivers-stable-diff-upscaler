# RiversHaveWings/Stable_Diffusion_Upscaler Cog model

This is an implementation of RiversHaveWings [StableDiffusionUpscaler](https://colab.research.google.com/drive/1o1qYJcFeywzCIdkfKJy7cTpgZTCM2EI4) as a Cog model. [Cog packages machine learning models as standard containers.](https://github.com/replicate/cog)

First, download the pre-trained weights:

    cog run script/download-weights

Then, you can run predictions:

    cog predict -i image=@demo.jpeg

## Sample

Demo Stable Diffusion image before upscaling:
![alt text](demo.jpeg)

Demo Stable Diffusion image after upscaling:
![alt text](output.png)
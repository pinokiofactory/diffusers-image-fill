## Diffusers Image Fill

Remove or replace objects from any image, on your local computer.

This is a very simple gradio app that lets you remove or replace objects from ANY image.

**Note**: that there are pretty good object removal AIs already.

The difference is that it's based on Diffusion and actually uses inpainting to do the removal.

Additional features I added to the original app.

Added features:

1. No square crop
2. Use Full resolution (default: 1024x1024)
3. Replace objects (via custom prompt)
4. Custom steps/guidance scale

This app uses *RealVisXL V5.0 Lightning* (based on Stable Diffusion XL) for inpainting.

This is already great. But just imagine the same thing, but using Flux. Imagine being able to:

1. Replace TEXT into images,
2. Render hi-def output
3. No hand issues

One cool thing about this app is, you can keep retrying until you find the one that you like.

Works on ALL OS (Mac, Windows, Linux)

Available on pinokio and Docker

### Install using Docker

Open `docker-compose.yml` and delete everything below the comment line if you don't have a `traefik` reverse proxy.

Then simply run:
```
docker compose up -d --build --force-recreate --remove-orphans
```

Check logs:
```
docker logs -f diffusers-image-fill
```

<h2 align="center">Stable Diffusion WebUI Forge - Neo (Docker)</h2>

> [!Warning]
> Requires an **NVIDIA** GPU<br>
> Ensure driver is up to date (`560+` required)

<hr>

## Unraid Deployment

<table>
	<tr>
		<th>Container Path</th>
		<th>Purpose</th>
	</tr>
	<tr>
		<td>
			<code>/home/forge/sd-webui/models</code>
		</td>
		<td>Checkpoint, Text Encoder, VAE, LoRA, ControlNet</td>
	</tr>
	<tr>
		<td>
			<code>/home/forge/sd-webui/output</code>
		</td>
		<td>Generated Images</td>
	</tr>
	<tr>
		<td>
			<code>/home/forge/sd-webui/extensions</code>
		</td>
		<td>User-Installed Extensions</td>
	</tr>
</table>

- The container runs as **UID 99** / **GID 100** (`nobody:users`) to match Unraid's default share permissions

<hr>

## Pre-built Image

A pre-built image with this code is maintained on Docker Hub by [@oromis995](https://github.com/oromis995):

```bash
docker pull oromis995/sd-forge-neo:latest
```

<hr>

## Building Locally

```bash
git clone https://github.com/Haoming02/sd-webui-forge-classic sd-webui-forge-neo --branch neo
cd sd-webui-forge-neo/docker
docker build -t forge-neo-local .
```

<hr>

## Image details

<table>
	<tr>
		<td>Base</td>
		<td><code>nvidia/cuda:12.6.3-runtime-ubuntu22.04</code></td>
	</tr>
	<tr>
		<td>Python</td>
		<td><code>3.13</code> via <b>uv</b></td>
	</tr>
	<tr>
		<td>PyTorch</td>
		<td>Latest (<code>cu126</code>)</td>
	</tr>
	<tr>
		<td>User</td>
		<td><code>forge</code> (UID 99 / GID 100)</td>
	</tr>
	<tr>
		<td>Port</td>
		<td>7860</td>
	</tr>
</table>

> [!Note]
> On the first run, `prepare_environment()` will install requirements and dependencies. This may take a few minutes

# Web demo / 网页演示

The demo runs the fixed E1 workflow on a ZIP of grayscale PNG sections. The browser shows matching original and restored XY, XZ and YZ views. A single index controls both images. The download contains the restored PNG stack and its processing record.

The ZIP may contain the PNGs at its root or together in one folder. Use a consistent numbering pattern such as `slice_00001.png`. Supply 16–128 usable 8-bit grayscale sections of the same dimensions. Each side must be 32–512 pixels and divisible by eight. The compressed ZIP and the sum of the unpacked PNG files each have a 100 MiB limit.

## Run on your computer

From the repository root, use Python 3.10 or newer. The demo uses the bundled E1 weights.

```bash
python -m pip install -e '.[demo]'
python -m demo.app
```

Open [http://127.0.0.1:7860](http://127.0.0.1:7860), drop a ZIP, then select **Process volume**. The plane selector and section slider show native pixels on the same grayscale range. **Processing** measures the fixed workflow including input validation, inference, postprocessing and saving the final images. **Server total** also includes reading the ZIP and assembling the views and download. Network upload and queue time are outside these measurements.

The app runs one restoration job at a time and keeps up to four queued requests. Jobs have isolated temporary folders. Results and the browser file cache are eligible for deletion after one hour. Download the output before leaving the page.

## Publish a URL

- **Lab server.** Install the CUDA-enabled PyTorch build for that server before installing `.[demo]`. Run `VEM_DEVICE=cuda python -m demo.app` from the repository root as a service. Ask the server administrator for a public HTTPS hostname and proxy it to `127.0.0.1:7860`. Configure the proxy to allow 100 MiB uploads plus multipart overhead and to forward streaming responses. [Gradio reverse proxy guide](https://www.gradio.app/guides/running-gradio-on-your-web-server-with-nginx).
- **Hugging Face Space.** Create a public Gradio Space on your account. Copy this release into the Space repository, add the two files below at its root and set `VEM_HOST=0.0.0.0` in Space Variables. The default device is CPU. Set `VEM_DEVICE=cuda` only on a GPU instance with CUDA-enabled PyTorch. Check the account and compute requirements before creating a Space. [Hugging Face Spaces](https://huggingface.co/docs/hub/spaces-overview).

For a Gradio Space, put this front matter at the top of the Space's `README.md`.

```yaml
---
title: vEMDestripe
sdk: gradio
sdk_version: 6.31.0
python_version: "3.11"
app_file: app.py
---
```

Add `app.py` at the Space root.

```python
from demo.app import main

if __name__ == "__main__":
    main()
```

The repository's `requirements.txt` supplies inference dependencies. Once the Space builds, use its public URL to test the ZIP upload and download before linking it from the manuscript.

## 中文

下载仓库，在根目录运行上面的两条命令，浏览器打开本机地址即可试用。上传同一体数据的 16–128 张同尺寸灰度 PNG，切片统一编号。处理后可用同一个滑块比较原图和结果，切换 XY、XZ、YZ，下载最终切片和处理记录。页面显示处理时间以及含解包和导出的服务端用时。

长期公开访问需要实验室服务器的 HTTPS 地址，或由作者账户创建并配置公开 Space。请先选定可用的主机。主机和网页运行所需的配置见上文。

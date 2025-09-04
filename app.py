import os
os.environ["NUMBA_CACHE_DIR"] = "/tmp"
os.environ["U2NET_HOME"] = os.path.join(os.getcwd(), "model")   # 👈 add this

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from rembg import remove, new_session
from PIL import Image
import io
import traceback
import uvicorn

app = FastAPI()

# adjust in production to only allow your frontend origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_EDGE = 1280  # server-side resize max edge (tweak for speed/quality)

# ✅ Create session once at startup
session = new_session("isnet-general-use")


def downscale_bytes(img_bytes: bytes, max_edge=MAX_EDGE):
    buf = io.BytesIO(img_bytes)
    img = Image.open(buf).convert("RGBA")
    w, h = img.size
    scale = max(w, h) / max_edge if max(w, h) > max_edge else 1.0
    if scale <= 1.0:
        buf.seek(0)
        return img, img.size, img_bytes, 1.0
    new_size = (int(w / scale), int(h / scale))
    small = img.resize(new_size, Image.LANCZOS)
    out_buf = io.BytesIO()
    small.save(out_buf, format="PNG")
    return small, (w, h), out_buf.getvalue(), scale


@app.post("/remove-bg")
async def remove_bg(file: UploadFile = File(...)):
    try:
        input_bytes = await file.read()

        # 1) downscale for faster processing
        small_img, orig_size, small_bytes, scale = downscale_bytes(input_bytes)

        # 2) run background removal using preloaded session
        out_bytes = remove(small_bytes, session=session)

        # 3) open result, optionally upscale to original size
        out_img = Image.open(io.BytesIO(out_bytes)).convert("RGBA")
        if scale > 1.0:
            out_img = out_img.resize(orig_size, Image.LANCZOS)

        buf = io.BytesIO()
        out_img.save(buf, format="PNG")
        result_bytes = buf.getvalue()

        return Response(content=result_bytes, media_type="image/png")
    except Exception as e:
        traceback.print_exc()
        return {"error": "processing_failed", "detail": str(e)}

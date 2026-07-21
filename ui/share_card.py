"""9:16 phone-frame render card for generated images.

Renders the output inside a mobile-phone frame (CSS) at a strict 9:16 aspect
ratio, mimicking a TikTok/Reels/Shorts canvas, and offers quick actions:
Download, Post to X, Post to Instagram.

Notes on "post to X/Instagram":
- X (Twitter) supports a Web Intent URL that pre-fills tweet text; images
  cannot be attached via a plain URL, so the user attaches the downloaded
  image after the composer opens. This is the only client-side-safe option
  without app credentials.
- Instagram has no public web intent for posting from desktop web; the
  reliable path is "download, then upload". We surface a download button and
  an Open Instagram link.
"""

from __future__ import annotations

import base64
import io
import urllib.parse
from typing import Optional

import streamlit as st
from PIL import Image


# The phone frame is rendered as a plain HTML/CSS fragment (a <div>, not an
# <iframe>). A full iframe is unnecessary: the frame is pure presentational CSS
# around an inline base64 image, and wrapping it in an <iframe> forced the ENTIRE
# document (CSS + chrome + image) to be re-serialized and re-embedded on every
# preview. Using st.markdown with a stable `key` lets Streamlit update the SAME
# element in place — we only swap the inner <img> data-URI, the phone chrome
# stays put. (The old claim that st.markdown/st.html strip data-URI images is
# false; verified that data-URI <img> and inline <style> both survive when
# rendered through st.markdown(unsafe_allow_html=True).)
_PHONE_CSS = """
<style>
  * { box-sizing: border-box; }
  .vd-wrap { display: flex; justify-content: center; padding: 6px 0 24px; overflow: hidden; }
  .vd-phone {
    position: relative;
    width: 384px;
    height: 684px;
    border-radius: 44px;
    padding: 12px;
    background: linear-gradient(160deg, #33363f 0%, #0f1015 100%);
    box-shadow: 0 10px 10px rgba(0,0,0,.6), inset 0 0 0 2px #44475a,
                inset 0 0 0 6px #16171d;
  }
  .vd-phone::before {  /* dynamic-island / notch */
    content: "";
    position: absolute;
    top: 20px; left: 50%; transform: translateX(-50%);
    width: 86px; height: 22px; border-radius: 14px;
    background: #05060a; z-index: 3;
  }
  .vd-side {  /* power button */
    position: absolute; right: -3px; top: 132px;
    width: 3px; height: 58px; border-radius: 3px;
    background: #23252e;
  }
  .vd-side.vol { left: -3px; right: auto; top: 108px; height: 40px; }
  .vd-side.vol2 { left: -3px; right: auto; top: 156px; height: 40px; }
  .vd-screen {
    position: relative;
    width: 100%; height: 100%;
    border-radius: 33px;
    overflow: hidden;
    background: #000;
  }
  .vd-screen img {
    width: 100%; height: 100%;
    object-fit: cover; display: block;
  }
  .vd-badge {
    position: absolute; bottom: 14px; left: 12px;
    padding: 3px 10px; border-radius: 999px;
    font: 700 11px/1.2 -apple-system, system-ui, sans-serif; color: #fff;
    background: rgba(0,0,0,.5); backdrop-filter: blur(4px); z-index: 2;
  }
  .vd-bar {  /* home indicator */
    position: absolute; bottom: 7px; left: 50%; transform: translateX(-50%);
    width: 96px; height: 5px; border-radius: 3px;
    background: rgba(255,255,255,.75); z-index: 2;
  }
  .vd-placeholder {  /* gradient shown until a render replaces it */
    position: absolute; inset: 0;
    background: linear-gradient(160deg, #7c3aed 0%, #a855f7 40%, #ec4899 100%);
    display: flex; align-items: center; justify-content: center;
    color: rgba(255,255,255,.9);
    font: 600 13px/1.4 -apple-system, system-ui, sans-serif;
    text-align: center; padding: 0 24px;
  }
</style>
"""

_PLACEHOLDER_TEXT = "Your 9:16 render will appear here"


def image_to_data_uri(img: Image.Image, fmt: str = "PNG") -> str:
    """Encode a PIL image as a base64 data URI for inline HTML embedding."""
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    mime = "image/png" if fmt.upper() == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{b64}"


def _png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _fit_9x16(img: Image.Image, size: Optional[tuple] = None) -> Image.Image:
    """Center-crop/pad the image to a 9:16 canvas for export.

    By default ``size`` is the image's own dimensions, so the export preserves
    the exact resolution that was generated (e.g. 768x1368) — no upscaling. Pass
    an explicit ``size`` only if you want a different target canvas.
    """
    target_w, target_h = size if size else img.size
    src_w, src_h = img.size
    scale = max(target_w / src_w, target_h / src_h)
    resized = img.resize((int(src_w * scale), int(src_h * scale)), Image.Resampling.LANCZOS)
    left = (resized.width - target_w) // 2
    top = (resized.height - target_h) // 2
    return resized.crop((left, top, left + target_w, top + target_h))


def _phone_html(img: Optional[Image.Image], caption: str = "") -> str:
    """Build the phone-frame HTML fragment (a <div>, not a full document).

    With ``img`` it embeds the image; without it, a purple->pink gradient
    placeholder is shown so the frame is always visible.
    """
    badge = f'<span class="vd-badge">{caption}</span>' if caption else ""
    if img is not None:
        data_uri = image_to_data_uri(img, "JPEG" if img.mode == "RGB" else "PNG")
        screen_html = f'<img src="{data_uri}" alt="preview">'
    else:
        screen_html = f'<div class="vd-placeholder">{_PLACEHOLDER_TEXT}</div>'
    return (
        _PHONE_CSS
        + '<div class="vd-wrap"><div class="vd-phone">'
        '<span class="vd-side vol"></span><span class="vd-side vol2"></span>'
        '<span class="vd-side"></span>'
        '<div class="vd-screen">'
        f"{screen_html}{badge}"
        '<div class="vd-bar"></div>'
        "</div></div></div>"
    )


def render_phone_frame(img: Optional[Image.Image] = None, caption: str = "",
                       target: Optional[object] = None) -> None:
    """Render just the iPhone frame (image or gradient placeholder), no buttons.

    Call with ``img=None`` to show the always-visible placeholder frame before a
    generation, and again with the result to swap in the render. ``target`` is an
    optional ``st.empty()`` placeholder; when given, the frame is written into it
    (so repeated previews REPLACE in place instead of stacking). With no target,
    it renders at the current Streamlit context.

    The frame is a plain HTML/CSS <div> rendered via st.markdown
    (unsafe_allow_html=True) — NOT an <iframe>. Unlike the old iframe approach
    the chrome and image are one document, so previews (driven by the parent's
    st.empty() placeholder) update the SAME element in place; we only resend the
    inner <img>, the phone chrome is not rebuilt. The data-URI <img> and inline
    <style> are both preserved by Streamlit's markdown renderer.
    """
    html = _phone_html(img, caption)
    if target is not None:
        target.markdown(html, unsafe_allow_html=True)
    else:
        st.markdown(html, unsafe_allow_html=True)


def render_share_card(img: Image.Image, caption: str = "", share_text: str = "",
                      key_prefix: str = "share", target: Optional[object] = None,
                      size: Optional[tuple] = None) -> None:
    """Render the 9:16 phone-frame preview and share/download actions.

    ``img`` is any PIL image. It is displayed cover-cropped inside a CSS iPhone
    frame (a plain HTML <div>, no iframe) sized purely for on-screen display
    (384x684). The download is exported at ``size`` (defaults to ``img``'s own
    resolution, e.g. 768x1368) so the saved file matches exactly what was
    generated — no upscaling to a fixed 1080x1920. ``target`` is an optional
    ``st.empty()`` placeholder; when given, everything is written into it so the
    card replaces the preview in place.

    IMPORTANT: when ``target`` is an ``st.empty()`` placeholder we must emit every
    element via the ``target.*`` methods (``target.markdown``,
    ``target.columns`` …) and NEVER wrap it in ``with target:``. Entering
    ``with target:`` REPLACES everything already written into the placeholder with
    the block's children, which would wipe the phone-frame markup and leave only
    the trailing caption. Bare ``st.markdown`` inside such a block also does not
    persist, so we always use the explicit ``target.`` form.
    """
    if target is None:
        target = st.container()
    # Frame first, written via the explicit target.markdown path that the previews
    # use (and that reliably persists inside an st.empty() placeholder).
    render_phone_frame(img, caption, target=target)
    # Buttons + tip go into a FRESH child container. An st.empty() is a single-slot
    # placeholder: calling target.columns(...) / target.download_button(...) on the
    # SAME DG slot REPLACES the frame we just wrote. Routing them through a child
    # container keeps the frame intact and appends the actions beneath it.
    actions = target.container()
    export = _fit_9x16(img, size)
    c1, c2, c3 = actions.columns(3)
    with c1:
        c1.download_button(
            "⬇️ Download 9:16",
            data=_png_bytes(export),
            file_name="visualdiffusion_9x16.png",
            mime="image/png",
            use_container_width=True,
            key=f"{key_prefix}_dl",
        )
    with c2:
        text = urllib.parse.quote(share_text or "Made with Visual Diffusion")
        x_url = f"https://twitter.com/intent/tweet?text={text}"
        c2.link_button("𝕏 Post to X", x_url, use_container_width=True)
    with c3:
        c3.link_button("📸 Instagram", "https://www.instagram.com/", use_container_width=True)
    actions.caption("Tip: download the 9:16 image, then attach it in the X composer / upload to Instagram.")

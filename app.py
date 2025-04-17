
import streamlit as st
import fitz, io, re, difflib, unicodedata, tempfile, os
from pathlib import Path
from comparison_algorithm import compare_pdf_first
from text_extraction import extract_text_from_word, extract_text_from_pdf_with_page_info
import pandas as pd
from collections import defaultdict
# ----------------------------------------------------------
st.set_page_config("期刊比對系統", "📊", layout="wide")
st.markdown("<h1 style='text-align:center;'>期刊比對系統</h1>", unsafe_allow_html=True)

MAX_PAGES = 20
def pdf_page_count(file):
    pos = file.tell()
    file.seek(0)
    data = file.read()
    file.seek(pos)
    with fitz.open(stream=data, filetype="pdf") as doc:
        return doc.page_count

def build_sub_pdf(file, pages:list[int])->io.BytesIO:
    pos=file.tell(); file.seek(0); data=file.read(); file.seek(pos)
    src=fitz.open(stream=data, filetype="pdf")
    new=fitz.open()
    for p in pages:
        if 1<=p<=src.page_count:
            new.insert_pdf(src, from_page=p-1, to_page=p-1)
    buf=io.BytesIO(new.tobytes())
    buf.seek(0)
    return buf

def page_image_bytes(pdf_bytes, page_num, zoom=1.5):
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        pix = doc.load_page(page_num-1).get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        return pix.tobytes("png")

# ----------------------------------------------------------
if "selected_pages" not in st.session_state:
    st.session_state.selected_pages=None
if "total_pages" not in st.session_state:
    st.session_state.total_pages=None

# -------- Sidebar -------------
with st.sidebar:
    st.header("⚙️ 比對設定")
    mode=st.radio("比對模式", ["Hybrid","Exact","Semantic","AI"], horizontal=True)
    threshold=st.slider("相似度閾值",0.0,1.0,0.6,0.05)

# -------- Upload --------------
c1,c2=st.columns(2)
with c1: word_file = st.file_uploader("上傳 Word (.docx)", type=["docx"])
with c2: pdf_file  = st.file_uploader("上傳 PDF (.pdf)", type=["pdf"])

# -------- Page selection ------
if pdf_file:
    st.session_state.total_pages = pdf_page_count(pdf_file)
    if st.session_state.total_pages>MAX_PAGES and st.session_state.selected_pages is None:
        st.warning(f"此 PDF 共 **{st.session_state.total_pages}** 頁，系統一次最多比對 {MAX_PAGES} 頁。請先選擇頁面。")
        sel_mode = st.radio("選擇方式", ["連續頁區間","指定頁碼"])
        if sel_mode=="連續頁區間":
            s,e=st.columns(2)
            start=s.number_input("起始頁",1,st.session_state.total_pages,1,1)
            end=e.number_input("結束頁",start,min(start+MAX_PAGES-1,st.session_state.total_pages),start+MAX_PAGES-1,1)
            pages=list(range(int(start),int(end)+1))
        else:
            pages=st.multiselect("選擇頁碼 (最多20)", list(range(1,st.session_state.total_pages+1)))
        if st.button("✅ 確定頁面") and pages and 1<=len(pages)<=MAX_PAGES:
            st.session_state.selected_pages=sorted(set(pages))
            st.success(f"已選擇：{st.session_state.selected_pages}")
    if st.session_state.selected_pages:
        st.info(f"將比對 PDF 頁面：{st.session_state.selected_pages}")

# -------- Reset selection -----
def reset_pages():
    st.session_state.selected_pages=None

# -------- Compare -------------
st.divider()
if st.button("🚀 開始比對"):
    if not word_file or not pdf_file:
        st.error("請同時上傳 Word 與 PDF")
        st.stop()
    # prepare pdf bytes
    if st.session_state.total_pages>MAX_PAGES and not st.session_state.selected_pages:
        st.error("請先選擇欲比對的 PDF 頁面")
        st.stop()
    if st.session_state.selected_pages:
        sub_pdf = build_sub_pdf(pdf_file, st.session_state.selected_pages)
        selected_pages = st.session_state.selected_pages
    else:
        pdf_file.seek(0)
        sub_pdf = io.BytesIO(pdf_file.read())
        sub_pdf.seek(0)
        selected_pages = list(range(1, pdf_page_count(pdf_file)+1))
    # extract
    word_data = extract_text_from_word(word_file)
    pdf_paragraphs = extract_text_from_pdf_with_page_info(sub_pdf)
    pdf_data={"paragraphs":pdf_paragraphs,"tables":[]}
    # compare
    res = compare_pdf_first(word_data, pdf_data,
                            comparison_mode=mode.lower(),
                            similarity_threshold=threshold,
                            ignore_options=dict(ignore_space=True,ignore_punctuation=True,
                                                ignore_case=True,ignore_newline=True))
    st.success(f"匹配 {res['statistics']['matched']} 段 / PDF {res['statistics']['total_pdf']} 段")
    # group by page
    page_map=defaultdict(list)
    for m in res['matches']:
        page_map[m['pdf_page']].append(m)
    # iterate pages
    pdf_bytes = sub_pdf.getvalue()
    for p in selected_pages:
        st.markdown(f"### 📄 PDF 頁 {p}")
        # show image
        img_bytes = page_image_bytes(pdf_bytes,p,zoom=0.8)
        st.image(img_bytes, use_column_width=True)
        # show matches table
        if p in page_map:
            df=pd.DataFrame([{
                "Word段":m["word_index"],
                "相似度":f"{m['similarity']:.2f}"
            } for m in page_map[p]])
            st.dataframe(df,use_container_width=True)
            for m in page_map[p]:
                with st.expander(f"Word 段 {m['word_index']} ({m['similarity']:.2f})"):
                    st.markdown("**Word：**")
                    st.write(m['word_text'])
                    st.markdown("**PDF 片段：**")
                    st.write(m['pdf_text'])
                    st.markdown("**差異：**",unsafe_allow_html=True)
                    st.markdown(m['diff_html'],unsafe_allow_html=True)
        else:
            st.info("此頁無匹配段落")
    st.button("🔄 重新選擇 PDF 頁面", on_click=reset_pages)

import pandas as pd
import streamlit as st
from typing import Dict, Any, List

def create_styled_table_diff(pdf_table: Dict[str, Any], 
                           word_table: Dict[str, Any], 
                           diff_info: Dict[str, Any]) -> pd.DataFrame:
    """創建帶有差異標示的表格視圖"""
    # 將表格資料轉換為DataFrame
    pdf_df = pd.DataFrame(pdf_table["data"])
    word_df = pd.DataFrame(word_table["data"])
    
    # 使用欄位映射對齊欄位
    mapping = diff_info["column_alignment"]["column_mapping"]
    word_df = word_df.rename(
        columns={v: k for k, v in mapping.items() if v}
    )[pdf_df.columns]
    
    # 計算差異掩碼
    diff_mask = (pdf_df != word_df) & ~(pdf_df.isna() & word_df.isna())
    
    # 添加行相似度
    pdf_df["行相似度"] = [
        f"{s['similarity']*100:.0f}%" 
        for s in diff_info["row_similarities"]
    ]
    
    # 應用樣式
    styled_df = pdf_df.style.apply(
        lambda row: [
            "background-color: #fff2ac" if diff_mask.loc[row.name, col] else "" 
            for col in pdf_df.columns[:-1]
        ], 
        axis=1
    )
    
    return styled_df

def display_table_comparison(table_match: Dict[str, Any], 
                           pdf_data: Dict[str, Any],
                           word_data: Dict[str, Any]):
    """顯示表格比對結果"""
    # 顯示表格對應關係與相似度
    st.markdown(
        f"**PDF表格 (頁{table_match['pdf_page']} #{table_match['pdf_index']}) "
        f"↔ Word表格 (#{table_match['word_index']})** "
        f"– 相似度: {table_match['similarity']*100:.0f}%"
    )
    
    # 取得對應的表格資料
    pdf_table = next(
        t for t in pdf_data["tables"] 
        if t["page"] == table_match["pdf_page"] 
        and t["index"] == table_match["pdf_index"]
    )
    word_table = next(
        t for t in word_data["tables"] 
        if t["index"] == table_match["word_index"]
    )
    
    # 創建並顯示帶差異標示的表格
    styled_df = create_styled_table_diff(
        pdf_table, 
        word_table, 
        table_match["diff"]
    )
    st.dataframe(styled_df, use_container_width=True)
    
    # 顯示欄位差異資訊
    diff = table_match["diff"]
    if diff["column_alignment"]["name_differences"]:
        st.write(
            "欄位名稱差異: " + 
            ", ".join([
                f"{pdf_col} ↔ {word_col}" 
                for pdf_col, word_col in diff["column_alignment"]["name_differences"]
            ])
        )
    if diff["column_alignment"]["order_different"]:
        st.write("⚠️ 欄位順序不同")
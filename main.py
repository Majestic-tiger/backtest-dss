import streamlit as st
from ui.common import render_navigation

st.set_page_config(page_title="dongpa", layout="wide")
render_navigation()
st.title("dongpa")
st.caption("동파법 LOC 백테스트 & 주문 스케줄러")
st.markdown("왼쪽 사이드바에서 페이지를 선택하세요.")

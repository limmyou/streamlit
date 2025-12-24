# -*- coding: utf-8 -*-
import streamlit as st
import requests
import psycopg2
import psycopg2.extras
import pytz
from datetime import datetime
import pandas as pd
import io
from PIL import Image
import zipfile
from st_aggrid import AgGrid, GridOptionsBuilder

# ======================================
#  FastAPI URL
# ======================================
FASTAPI_URL_DETECT = "https://fastapi-production-c437.up.railway.app/detect"
FASTAPI_URL_SEGMENT = "https://fastapi-production-c437.up.railway.app/segment"
FASTAPI_STATUS_URL = "https://fastapi-production-c437.up.railway.app/status"

# ======================================
#  PostgreSQL 연결
# ======================================
def get_connection():
    return psycopg2.connect(
        host=st.secrets["POSTGRES_HOST"],
        user=st.secrets["POSTGRES_USER"],
        password=st.secrets["POSTGRES_PASSWORD"],
        database=st.secrets["POSTGRES_DATABASE"],
        port=int(st.secrets["POSTGRES_PORT"]),
        cursor_factory=psycopg2.extras.DictCursor
    )

# ======================================
#  사용자 로그 저장 (YOLO)
# ======================================
def save_user_log(name, timestamp):
    try:
        conn = get_connection()
        cursor = conn.cursor()
        query = "INSERT INTO user_log_yolo (name, access_time) VALUES (%s, %s)"
        cursor.execute(query, (name, timestamp))
        conn.commit()
        conn.close()
    except Exception as e:
        st.error(f"[사용자 로그 저장 실패] {e}")


# ======================================
#  이미지 데이터 저장
# ======================================
def insert_to_image_data(name, filename, info_value, filesize, inference_time, timestamp, mode):
    try:
        conn = get_connection()
        cursor = conn.cursor()

        if mode == "detect":  # YOLO
            query = """
                INSERT INTO image_data_yolo 
                (name, filename, object_count, filesize, inference_time, timestamp)
                VALUES (%s, %s, %s, %s, %s, %s)
            """
            params = (name, filename, info_value, filesize, inference_time, timestamp)

        else:  # DeepLab
            query = """
                INSERT INTO image_data_dlab 
                (name, filename, area_count, inference_time, timestamp)
                VALUES (%s, %s, %s, %s, %s)
            """
            params = (name, filename, info_value, inference_time, timestamp)

        cursor.execute(query, params)
        conn.commit()
        conn.close()
        return True

    except Exception as e:
        st.error(f"[DB 저장 실패] {e}")
        return False


# ======================================
#  Streamlit 설정
# ======================================
st.set_page_config(page_title="이미지 분석 시스템", layout="wide")
st.markdown("<h1 style='text-align:center;'>이미지 분석 시스템</h1>", unsafe_allow_html=True)

kst = pytz.timezone("Asia/Seoul")

# ======================================
#  탭 구성
# ======================================
tab1, tab2 = st.tabs(["이미지 업로드", "DB 결과 조회"])


# ======================================
# TAB 1 - 이미지 업로드
# ======================================
with tab1:
    name = st.text_input("사용자 이름")

    # 첫 접속: 사용자 로그 저장
    if name and "saved_name" not in st.session_state:
        timestamp = datetime.now(kst).replace(tzinfo=None)
        save_user_log(name, timestamp)
        st.session_state.saved_name = name
        st.success(f"'{name}' 접속 기록 저장 완료")

    model_choice = st.radio("모델 선택", ["객체 탐지 (YOLO)", "면적 분할 (DeepLabV3+)"])

    uploaded_file = st.file_uploader("이미지 또는 ZIP 파일 업로드", type=["jpg", "jpeg", "png", "zip"])

    if uploaded_file and not name:
        st.warning("이름을 먼저 입력해주세요.")
        st.stop()

    if uploaded_file and name:

        # ★ 중복 저장 방지 (파일명 + 사용자명 기반)
        file_key = f"{name}_{uploaded_file.name}"
        if st.session_state.get(file_key):
            st.stop()
        st.session_state[file_key] = True

        timestamp = datetime.now(kst).replace(tzinfo=None)

        api_url = FASTAPI_URL_DETECT if model_choice == "객체 탐지 (YOLO)" else FASTAPI_URL_SEGMENT

        # 상태 체크 (YOLO 전용)
        try:
            status_response = requests.get(FASTAPI_STATUS_URL)
            if status_response.ok and status_response.json().get("status") == "busy":
                st.warning("모델이 추론 중입니다.")
                st.stop()
        except:
            pass

        # ===========================
        # ZIP 파일 처리
        # ===========================
        if uploaded_file.name.lower().endswith(".zip"):
            results = []

            try:
                with zipfile.ZipFile(uploaded_file) as z:
                    image_infos = [info for info in z.infolist()
                                   if info.filename.lower().endswith((".jpg", ".jpeg", ".png"))]

                    if not image_infos:
                        st.warning("압축 파일에 유효한 이미지가 없습니다.")
                        st.stop()

                    for info in image_infos:

                        zip_key = f"{name}_{info.filename}"
                        if st.session_state.get(zip_key):
                            continue
                        st.session_state[zip_key] = True

                        file_bytes = z.read(info)

                        try:
                            response = requests.post(api_url, files={"file": file_bytes})

                            if response.status_code == 200:
                                result = response.json()

                                if model_choice == "객체 탐지 (YOLO)":
                                    info_value = result.get("object_count", 0)
                                    mode = "detect"
                                else:
                                    info_value = result.get("area_cm2_assumed", 0)
                                    mode = "segment"

                                inference_time = result.get("inference_time_ms", 0) / 1000
                                filesize = uploaded_file.size

                                insert_to_image_data(
                                    name, info.filename, info_value, filesize,
                                    inference_time, timestamp, mode
                                )

                                results.append({"파일명": info.filename, "결과": info_value})

                            else:
                                results.append({"파일명": info.filename, "결과": "실패"})

                        except Exception as e:
                            st.warning(f"{info.filename} 처리 중 오류: {e}")

                if results:
                    st.dataframe(pd.DataFrame(results), use_container_width=True)

            except Exception as e:
                st.error(f"ZIP 처리 오류: {e}")

        # ===========================
        # 단일 이미지 처리
        # ===========================
        else:
            file_bytes = uploaded_file.getvalue()

            try:
                response = requests.post(api_url, files={"file": file_bytes})

                if response.status_code == 200:
                    result = response.json()
                    inference_time = result.get("inference_time_ms", 0) / 1000

                    if model_choice == "객체 탐지 (YOLO)":
                        info_value = result.get("object_count", 0)
                        st.success(f"탐지된 개체 수: {info_value}개")
                        mode = "detect"
                    else:
                        info_value = result.get("area_cm2_assumed", 0)
                        ratio = result.get("area_ratio_percent", 0)
                        st.success(f"면적: {info_value} cm² ({ratio:.2f}%)")
                        mode = "segment"

                    filesize = uploaded_file.size

                    insert_to_image_data(
                        name, uploaded_file.name, info_value, filesize,
                        inference_time, timestamp, mode
                    )

                    st.success("DB 저장 완료")

                else:
                    st.error(f"API 응답 실패 (Status {response.status_code})")

            except Exception as e:
                st.error(f"FastAPI 요청 실패: {e}")


# ======================================
# TAB 2 - DB 조회
# ======================================
with tab2:
    st.subheader("최근 기록 조회")

    db_choice = st.radio("조회할 모델 선택", ["객체 탐지 (YOLO)", "면적 분할 (DeepLabV3+)"])
    query_name = st.text_input("사용자 이름", key="query_name")

    try:
        conn = get_connection()

        # YOLO 조회
        if db_choice == "객체 탐지 (YOLO)":
            base_query = """
                SELECT name, filename, object_count, inference_time, timestamp
                FROM image_data_yolo
                ORDER BY timestamp DESC
                LIMIT 30
            """
            name_query = """
                SELECT name, filename, object_count, inference_time, timestamp
                FROM image_data_yolo
                WHERE name = %s
                ORDER BY timestamp DESC
            """
            column_names = ["사용자", "파일명", "객체 수", "추론 시간(초)", "업로드 시각"]

        # DLAB 조회
        else:
            base_query = """
                SELECT name, filename, area_count, inference_time, timestamp
                FROM image_data_dlab
                ORDER BY timestamp DESC
                LIMIT 30
            """
            name_query = """
                SELECT name, filename, area_count, inference_time, timestamp
                FROM image_data_dlab
                WHERE name = %s
                ORDER BY timestamp DESC
            """
            column_names = ["사용자", "파일명", "면적 개수", "추론 시간(초)", "업로드 시각"]

        df = (
            pd.read_sql(name_query, conn, params=[query_name])
            if query_name
            else pd.read_sql(base_query, conn)
        )
        conn.close()

        if not df.empty:
            df.columns = column_names
            st.dataframe(df, use_container_width=True)

            today_str = datetime.now(kst).strftime('%y%m%d')
            filename = f"results_{query_name or 'all'}_{today_str}.xlsx"

            # Excel 저장
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name="Results")

            st.download_button(
                "결과 다운로드",
                data=excel_buffer.getvalue(),
                file_name=filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        else:
            st.info("조회할 데이터가 없습니다.")

    except Exception as e:
        st.error(f"[DB 조회 실패] {e}")

# app.py
import streamlit as st
import pandas as pd
from utils.openai_helper import get_openai_streaming_response, get_openai_response
from helper.crawl_selenium import get_product_info, get_ebay_reviews, get_youtube_comments

def main():
    st.set_page_config(layout="wide", page_title="Multi-source Chatbot", page_icon="🤖")
    st.sidebar.header("Chatbot Configuration")

    # Khởi tạo session state
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []
    if 'messages_history' not in st.session_state:
        st.session_state.messages_history = [{
            'role': 'system',
            'content': 'Bạn là trợ lý AI hữu ích.'
        }]
    if 'current_items' not in st.session_state:
        st.session_state.current_items = []  # lưu reviews/comments vừa crawl

    # Chọn nguồn dữ liệu: Amazon, eBay hoặc YouTube
    source = st.sidebar.selectbox("Chọn nguồn dữ liệu", ["Amazon", "eBay", "YouTube"])
    url = st.sidebar.text_input("Nhập URL sản phẩm/video")
    num_comments = st.sidebar.slider("Số lượng comments/reviews", min_value=1, max_value=50, value=10)

    # Nút Scrape
    if st.sidebar.button("Scrape"):
        if not url:
            st.sidebar.error("Vui lòng nhập URL!")
        else:
            # Xóa sạch conversation cũ để chỉ hiển thị sản phẩm mới
            st.session_state.conversation = []

            with st.sidebar.status("Đang crawl dữ liệu..."):
                # Gọi hàm phù hợp theo nguồn
                if source == "Amazon":
                    data = get_product_info(url)
                    items = data.get('reviews', [])
                elif source == "eBay":
                    data = get_ebay_reviews(url, num_comments)
                    items = data.get('comments', [])
                else:
                    data = get_youtube_comments(url, num_comments)
                    items = data.get('comments', [])

                # Lưu dữ liệu vào session
                st.session_state.product_data = data
                st.session_state.current_items = items

                # Xuất dữ liệu thô ra CSV để lưu trữ
                df = pd.DataFrame(items)
                filename = f"{source.lower()}_data.csv"
                df.to_csv(filename, index=False, encoding='utf-8-sig')
                st.sidebar.success(f"Dữ liệu crawl đã được lưu vào {filename}")
                # Cho phép tải file CSV
                with open(filename, 'rb') as f:
                    st.sidebar.download_button(
                        label="Tải CSV",
                        data=f,
                        file_name=filename,
                        mime='text/csv'
                    )

                # Tóm tắt và lưu context chi tiết
                summary_prompt = f"Dưới đây là các nội dung người dùng từ {source}:\n"
                for i, it in enumerate(items, start=1):
                    summary_prompt += f"Review {i}: {it.get('text')}\n"
                summary = get_openai_response(summary_prompt)

                # Xây dựng messages_history chỉ chứa context mới
                system_content = f"Bạn là trợ lý AI. Dưới đây là tóm tắt nội dung từ {source}:\n{summary}\n" + summary_prompt
                st.session_state.messages_history = [
                    {'role': 'system', 'content': system_content}
                ]

                # Thêm tin nhắn khởi tạo vào conversation
                st.session_state.conversation.append((
                    'assistant',
                    f"Hoàn thành crawl {source}!\n{summary}"
                ))
            # Reload trang để cập nhật chat area
            st.rerun()

    # Chat area
    st.title("Multi-source Chatbot")

    # Hiển thị lịch sử chat
    for role, msg in st.session_state.conversation:
        if role == 'user':
            st.markdown(f"**Bạn:** {msg}")
        else:
            st.markdown(f"**Bot:** {msg}")

    # Input chat
    user_input = st.chat_input("Nhập câu hỏi...")
    if user_input:
        # Append user message and call OpenAI for any question (bao gồm hỏi số lượng)
        st.session_state.conversation.append(('user', user_input))
        st.session_state.messages_history.append({'role': 'user', 'content': user_input})

        full_resp = ''
        for chunk in get_openai_streaming_response(st.session_state.messages_history):
            full_resp += chunk

        st.session_state.conversation.append(('assistant', full_resp))
        st.session_state.messages_history.append({'role': 'assistant', 'content': full_resp})
        st.rerun()

if __name__ == '__main__':
    main()

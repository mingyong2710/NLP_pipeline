import streamlit as st
from utils.openai_helper import get_openai_streaming_response
from utils.openai_helper import get_openai_response
from helper.crawl_selenium import get_product_info

def main():
    st.set_page_config(layout="wide", page_title="Amazon Chatbot", page_icon="🤖")

    # Sidebar
    st.sidebar.header("Chatbot Configuration")

    # 1) Chọn mô hình
    selected_model = st.sidebar.selectbox(
        "Chọn Model",
        options=["GPT-4o-MINI", "GPT-4", "GPT-4 32k", "Model khác..."]
    )

    # 2) Nhập URL của sản phẩm Amazon
    product_url = st.sidebar.text_input("Nhập Amazon Product URL")

    # 3) Slider chọn số lượng review
    num_reviews = st.sidebar.slider("Số lượng review", min_value=1, max_value=50, value=10)

    # Khởi tạo session state cho product data
    if "product_data" not in st.session_state:
        st.session_state.product_data = None

    # 4) Nút scrape
    if st.sidebar.button("Scrape"):
        with st.sidebar.status("Đang scrape dữ liệu sản phẩm...") as status:
            # Hiển thị thông báo đang scrape
            st.sidebar.write(f"Đang xử lý scrape reviews từ: {product_url}")
            
            # Gọi hàm scrape để lấy thông tin sản phẩm
            product_data = get_product_info(product_url)
            
            # Lưu dữ liệu vào session state để sử dụng sau
            st.session_state.product_data = product_data
            
            # Giới hạn số lượng review theo người dùng chọn
            if "reviews" in product_data and len(product_data["reviews"]) > num_reviews:
                product_data["reviews"] = product_data["reviews"][:num_reviews]
            
            status.update(label="Đang tạo tóm tắt sản phẩm...", state="running")
            
            # Tạo prompt để sinh tóm tắt sản phẩm
            summary_prompt = f"""
            Hãy tạo một bản tóm tắt ngắn gọn về sản phẩm này dựa trên thông tin sau:
            
            Tên sản phẩm: {product_data.get('title', 'Không rõ')}
            Giá: {product_data.get('price', 'Không rõ')}
            Đánh giá: {product_data.get('rating', 'Không rõ')}
            Số lượng đánh giá: {product_data.get('review_count', 'Không rõ')}
            Mô tả: {product_data.get('description', 'Không rõ')}
            
            THÔNG TIN BẢNG CHI TIẾT SẢN PHẨM:
            {product_data.get('table_info', 'Không có thông tin chi tiết')}
            
            Tóm tắt nên bao gồm: Đây là sản phẩm gì, các tính năng chính, điểm mạnh, giá cả, lời khuyên mua hàng, v.v.
            """
            
            # Gọi OpenAI để tạo tóm tắt
            product_summary = get_openai_response(summary_prompt)
            
            # Tạo context từ reviews để thêm vào system message
            reviews_context = "Dưới đây là các đánh giá của người dùng về sản phẩm:\n\n"
            for i, review in enumerate(product_data.get("reviews", [])):
                reviews_context += f"Review #{i+1}:\n"
                reviews_context += f"- Tiêu đề: {review.get('title', 'Không có tiêu đề')}\n"
                reviews_context += f"- Người đánh giá: {review.get('author', 'Ẩn danh')}\n"
                reviews_context += f"- Nội dung: {review.get('text', 'Không có nội dung')}\n"
                reviews_context += f"- Ngày: {review.get('date', 'Không rõ ngày')}\n\n"
            
            # Cập nhật system message với thông tin sản phẩm và reviews
            system_message = f"""Bạn là trợ lý AI hữu ích, thân thiện và trung thực.
            
            THÔNG TIN SẢN PHẨM:
            Tên: {product_data.get('title', 'Không rõ')}
            Giá: {product_data.get('price', 'Không rõ')}
            Đánh giá: {product_data.get('rating', 'Không rõ')}
            Số lượng đánh giá: {product_data.get('review_count', 'Không rõ')}
            
            TÓM TẮT SẢN PHẨM:
            {product_summary}
            
            {reviews_context}
            
            Hãy sử dụng thông tin trên để trả lời các câu hỏi của người dùng về sản phẩm này.
            Khi được hỏi về đánh giá hoặc cảm nhận về sản phẩm, hãy dựa vào các đánh giá của người dùng đã cung cấp.
            Khi không có thông tin để trả lời, hãy thừa nhận rằng bạn không có đủ thông tin và không tự tạo ra thông tin giả.
            """
            
            # Cập nhật session state messages history
            if "messages_history" in st.session_state:
                # Tìm và thay thế system message cũ
                for i, msg in enumerate(st.session_state.messages_history):
                    if msg.get("role") == "system":
                        st.session_state.messages_history[i] = {"role": "system", "content": system_message}
                        break
                else:
                    # Nếu không tìm thấy system message, thêm vào đầu danh sách
                    st.session_state.messages_history.insert(0, {"role": "system", "content": system_message})
            else:
                st.session_state.messages_history = [{"role": "system", "content": system_message}]
            
            # Thêm tin nhắn tự động từ hệ thống để hiển thị tóm tắt sản phẩm
            st.session_state.conversation.append(("assistant", f"Đã scrape thông tin sản phẩm thành công!\n\n{product_summary}"))
            
            status.update(label="Hoàn tất!", state="complete")
        
        # Buộc Streamlit rerun để hiển thị thay đổi
        st.rerun()

    # ------------------------------------------------
    # Phần chính: giao diện chat
    # ------------------------------------------------

    st.title("Amazon Chatbot")

    # Khởi tạo session_state để lưu trữ lịch sử hội thoại
    if "conversation" not in st.session_state:
        st.session_state.conversation = []
    
    # Khởi tạo messages_history cho API
    if "messages_history" not in st.session_state:
        st.session_state.messages_history = [
            {"role": "system", "content": "Bạn là trợ lý AI hữu ích, thân thiện và trung thực."}
        ]

    # Hàm hiển thị một tin nhắn trong khu vực chat
    # role: 'user' hoặc 'assistant'
    # text: nội dung tin nhắn
    def display_message(role, text):
        # Người dùng (hiển thị bên phải, 70% độ rộng)
        if role == "user":
            st.markdown(
                f"""
                <div style='text-align: right; margin: 10px; display: flex; justify-content: flex-end; align-items: flex-start;'>
                    <div style='display: inline-block; background-color: #DCF8C6; padding: 8px 12px; border-radius: 8px; max-width: 70%; margin-right: 8px;'>
                        {text}
                    </div>
                    <div style='width: 36px; height: 36px; border-radius: 50%; background-color: #128C7E; color: white; display: flex; justify-content: center; align-items: center; font-weight: bold;'>
                        <span>👤</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        # Bot (hiển thị 100% độ rộng, bên dưới câu hỏi)
        else:
            st.markdown(
                f"""
                <div style='margin: 10px; display: flex; align-items: flex-start;'>
                    <div style='width: 36px; height: 36px; border-radius: 50%; background-color: #4285F4; color: white; display: flex; justify-content: center; align-items: center; margin-right: 8px; font-weight: bold;'>
                        <span>🤖</span>
                    </div>
                    <div style='display: inline-block; background-color: #F1F0F0; padding: 8px 12px; border-radius: 8px; max-width: 90%;'>
                        {text}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    # Hiển thị lịch sử các tin nhắn
    for role, content in st.session_state.conversation:
        display_message(role, content)

    # Input của chat
    user_input = st.chat_input("Nhập câu hỏi của bạn...")

    # Xử lý khi người dùng ấn Enter
    if user_input:
        # Thêm tin nhắn của user vào hội thoại hiển thị UI
        st.session_state.conversation.append(("user", user_input))
        
        # Thêm tin nhắn user vào history cho API
        st.session_state.messages_history.append({"role": "user", "content": user_input})
        
        # Hiển thị tin nhắn của người dùng ngay lập tức
        display_message("user", user_input)
        
        # Tạo placeholder để hiển thị tin nhắn đang typing
        typing_placeholder = st.empty()
        typing_placeholder.markdown(
            """
            <div style='margin: 10px; display: flex; align-items: flex-start;'>
                <div style='width: 36px; height: 36px; border-radius: 50%; background-color: #4285F4; color: white; display: flex; justify-content: center; align-items: center; margin-right: 8px; font-weight: bold;'>
                    <span>🤖</span>
                </div>
                <div style='display: inline-block; background-color: #F1F0F0; padding: 8px 12px; border-radius: 8px;'>
                    <i>Đang nhập...</i>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        try:
            # Gọi OpenAI API và hiển thị kết quả streaming với toàn bộ lịch sử
            full_response = ""
            for response_chunk in get_openai_streaming_response(st.session_state.messages_history):
                if response_chunk:
                    full_response += response_chunk
                    # Cập nhật phản hồi đang xây dựng
                    typing_placeholder.markdown(
                        f"""
                        <div style='margin: 10px; display: flex; align-items: flex-start;'>
                            <div style='width: 36px; height: 36px; border-radius: 50%; background-color: #4285F4; color: white; display: flex; justify-content: center; align-items: center; margin-right: 8px; font-weight: bold;'>
                                <span>🤖</span>
                            </div>
                            <div style='display: inline-block; background-color: #F1F0F0; padding: 8px 12px; border-radius: 8px; max-width: 90%;'>
                                {full_response}▌
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            
            # Xóa placeholder typing và thêm tin nhắn hoàn chỉnh vào hội thoại UI
            typing_placeholder.empty()
            st.session_state.conversation.append(("assistant", full_response))
            
            # Thêm phản hồi của bot vào history cho OpenAI API
            st.session_state.messages_history.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            # Xử lý lỗi nếu có
            error_message = f"Có lỗi xảy ra: {str(e)}"
            typing_placeholder.empty()
            st.session_state.conversation.append(("assistant", error_message))
            display_message("assistant", error_message)

        # Reset ô input về rỗng để sẵn sàng cho câu hỏi tiếp theo
        st.rerun()

if __name__ == "__main__":
    main()
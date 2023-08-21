import os
import torch
import cv2
import json
import shutil
import numpy as np
import streamlit as st
from PIL import Image
from pathlib import Path


@st.cache()
def load_model(path: str = 'weights/best.pt'):
    model = torch.hub.load('ultralytics/yolov5', 'custom', path = path)
    return model

@st.cache()
def load_file_structure(path: str = 'src/all_img.json') -> dict:
    with open(path, 'r') as f:
        return json.load(f)

@st.cache()
def load_list_of_images(
        all_images: dict,
        image_files_dtype: str,
        diseases_species: str
        ) -> list:
    species_dict = all_images.get(image_files_dtype)
    list_of_files = species_dict.get(diseases_species)
    return list_of_files

@st.cache(allow_output_mutation = True)
def get_prediction(img_bytes, model):
    results = model(img_bytes)  
    return results


def main():
    st.set_page_config(
        page_title = "Automatic Detection of Dermatological diseases",
        page_icon = "🔎",
        layout = "wide",
        initial_sidebar_state = "expanded"
    )

    model = load_model()
    all_images = load_file_structure()
    types_of_diseases = sorted(list(all_images['train'].keys()))
   
    dtype_file_structure_mapping = {
    'Images Used To Train The Model': 'train',
    'Images Used To Tune The Model': 'valid',
    'Images The Model Has Never Seen': 'test'
}

    data_split_names = list(dtype_file_structure_mapping.keys())


    with st.sidebar:
        st.image(Image.open('src/LOGOMINI.jpg'), width = 100)

        select_page = st.radio("CHỌN", ["TRANG CHỦ", "VỀ ADD", "LIÊN HỆ"])
        st.markdown("<br /><br /><br /><br /><br /><br />", unsafe_allow_html = True)
        st.markdown("<hr />", unsafe_allow_html = True)

    if select_page == "TRANG CHỦ":
        col1, col2 = st.columns([8.1, 4])
        file_img, file_vid, key_path, web_cam = '', '', '',''

        with col2:
            logo = Image.open('src/LOGOBIG.jpg')
            st.image(logo, use_column_width = True)

            with st.expander("Cách sử dụng ADD?", expanded = True):
                 with open('src/title/STORY.md', 'r', encoding='utf-8') as file:
                     markdown_text = file.read()

                  # Hiển thị nội dung markdown
            st.markdown(markdown_text, unsafe_allow_html=True)


        with col1:
            with open('src/title/INFO.md', 'r', encoding='utf-8') as file:
                 markdown_text = file.read()

                  # Hiển thị nội dung markdown
            st.markdown(markdown_text, unsafe_allow_html=True)

            choice_way = st.radio("Chọn một", ["Tải ảnh lên","Tải video lên","Sử dụng webcam","Chọn từ ảnh có sẵn"])

            if choice_way == "Tải ảnh lên":
                file_img = st.file_uploader('Tải một hình ảnh về bệnh da liễu')

                if file_img:
                    img = Image.open(file_img)

            elif choice_way == "Tải video lên":
                file_vid = st.file_uploader('Tải một video về bệnh da liễu')
                if file_vid:
                    frame_skip = 20 # display every 300 frames
                    st.video(file_vid)
                    vid = file_vid.name
                    with open(vid, mode='wb') as f:
                        f.write(file_vid.read()) # save video to disk
     

            elif choice_way == "Sử dụng webcam":
                frame_skip = 100
                st.write("Để dừng mở webcam vui lòng nhấn x trên bàn phím")
                web_cam = cv2.VideoCapture(0)
                vid_cod = cv2.VideoWriter_fourcc(*'XVID')
                output = cv2.VideoWriter("cam_video.mp4", vid_cod, 20.0, (640, 480))

                while True:
                    # Capture each frame of webcam video
                    ret, frame = web_cam.read()
                    if not ret:
                        print("Failed to read frame from webcam.")
                        break
                    
                    cv2.imshow("Webcam", frame)
                    output.write(frame)
                    
                    # Close and break the loop after pressing "x" key
                    if cv2.waitKey(1) & 0xFF == ord('x'):
                        break

                # close the already opened camera
                web_cam.release()
                # close the already opened file
                output.release()
                # close the window and de-allocate any associated memory usage
                cv2.destroyAllWindows()
                vid = 'cam_video.mp4'



            else:

                dataset_type = st.selectbox("Loại dữ liệu", data_split_names)
                data_folder = dtype_file_structure_mapping[dataset_type]

                selected_species = st.selectbox("Loại bệnh da liễu",types_of_diseases)
                available_images = load_list_of_images(all_images, data_folder, selected_species)
                image_name = st.selectbox("Tên hình ảnh", available_images)

                key_path = os.path.join('dermatological_diseases_dataset', data_folder, image_name)
                img = cv2.imread(key_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    

            recipe_button = st.button('Lấy kết quả!')

        st.markdown("<hr />", unsafe_allow_html = True)

        if recipe_button:
            with st.spinner("Chờ trong giây lát..."):
                if file_img or key_path:
                    col3, col4 = st.columns([5, 4])
                    with col3: 
                        if os.path.isdir('./runs'):
                            shutil.rmtree('./runs')

                        results = get_prediction(img, model)
                        results.save()

                        img_res = cv2.imread('./runs/detect/exp/image0.jpg')
                        img_res = cv2.cvtColor(img_res, cv2.COLOR_BGR2RGB)
                        st.header("Đây là kết quả phát hiện!")
                        st.image(img_res, use_column_width=True)

                        df = results.pandas().xyxy[0]
                        del df['class']
                        st.write(df)
                    with col4:
                        st.header("Mô tả")

                        des = set()
                        for name_type in df['name']:
                            if name_type not in des:
                                # Xử lý hiển thị mô tả cho từng loại bệnh
                                if name_type == 'muncoc':
                                       with st.expander("MỤN CÓC - U MỀM"):

                                            with open('src/title/MUNCOC.md', 'r', encoding='utf-8') as file:
                                                markdown_text = file.read()

                                            # Hiển thị nội dung  markdown
                                            st.markdown(markdown_text, unsafe_allow_html=True)

                                
                                    
                                elif name_type == 'vaynen':
                                    with st.expander("VẨY NẾN- Á SỪNG"):

                                        with open('src/title/VAYNEN.md', 'r', encoding='utf-8') as file:
                                            markdown_text = file.read()

                                            # Hiển thị nội dung  markdown
                                        st.markdown(markdown_text, unsafe_allow_html=True)
                                            

                                elif name_type == 'trungcado':
                                    with st.expander("TRỨNG CÁ ĐỎ"):

                                        with open('src/title/TRUNGCADO.md', 'r', encoding='utf-8') as file:
                                            markdown_text = file.read()

                                            # Hiển thị nội dung  markdown
                                        st.markdown(markdown_text, unsafe_allow_html=True)

                                            

                                elif name_type == 'hacto':
                                    with st.expander("UNG THƯ HẮC TỐ"):

                                        with open('src/title/HACTO.md', 'r', encoding='utf-8') as file:
                                            markdown_text = file.read()

                                            # Hiển thị nội dung  markdown
                                        st.markdown(markdown_text, unsafe_allow_html=True)

                                            
                                elif name_type == 'bachbien':
                                    with st.expander("BẠCH BIẾN"):

                                        with open('src/title/BACHBIEN.md', 'r', encoding='utf-8') as file:
                                            markdown_text = file.read()

                                            # Hiển thị nội dung  markdown
                                        st.markdown(markdown_text, unsafe_allow_html=True)

                                des.add(name_type)

                        if not des:
                            st.info("Không có dữ liệu để mô tả!")

                elif file_vid or web_cam:
                    vid_cap = cv2.VideoCapture(vid)
                    cur_frame = 0
                    success = True

                    while success:
                        ret, frame = vid_cap.read()  # get next frame from video
                        if not ret:
                            print("Failed to read frame from video.")
                            break
                        
                        if cur_frame % frame_skip == 0:  # only analyze every n=300 frames
                            print('frame: {}'.format(cur_frame))
                            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            pil_img = Image.fromarray(img)
                            
                            if os.path.exists('./runs'):
                                shutil.rmtree('./runs')
                            
                            results = get_prediction(pil_img, model)
                            results.save()

                            st.header("Đây là kết quả phát hiện!")

                            img_res = cv2.imread('./runs/detect/exp/image0.jpg')
                            if img_res is not None:
                                img_res = cv2.cvtColor(img_res, cv2.COLOR_BGR2RGB)
                                
                                col5, col6 = st.columns([5, 4])
                                with col5:
                                    st.image(img_res, use_column_width=True)

                                    df = results.pandas().xyxy[0]
                                    del df['class']
                                    st.write(df)
                                with col6:
                                    st.header("Mô tả")

                                    des = set()
                                    for name_type in df['name']:
                                        if name_type not in des:
                                            # Xử lý hiển thị mô tả cho từng loại bệnh
                                            if name_type == 'muncoc':
                                                with st.expander("MỤN CÓC - U MỀM"):

                                                        with open('src/title/MUNCOC.md', 'r', encoding='utf-8') as file:
                                                            markdown_text = file.read()

                                                        # Hiển thị nội dung  markdown
                                                        st.markdown(markdown_text, unsafe_allow_html=True)

                                            
                                                
                                            elif name_type == 'vaynen':
                                                with st.expander("VẨY NẾN- Á SỪNG"):

                                                    with open('src/title/VAYNEN.md', 'r', encoding='utf-8') as file:
                                                        markdown_text = file.read()

                                                        # Hiển thị nội dung  markdown
                                                    st.markdown(markdown_text, unsafe_allow_html=True)
                                                        

                                            elif name_type == 'trungcado':
                                                with st.expander("TRỨNG CÁ ĐỎ"):

                                                    with open('src/title/TRUNGCADO.md', 'r', encoding='utf-8') as file:
                                                        markdown_text = file.read()

                                                        # Hiển thị nội dung  markdown
                                                    st.markdown(markdown_text, unsafe_allow_html=True)

                                                        

                                            elif name_type == 'hacto':
                                                with st.expander("UNG THƯ HẮC TỐ"):

                                                    with open('src/title/HACTO.md', 'r', encoding='utf-8') as file:
                                                        markdown_text = file.read()

                                                        # Hiển thị nội dung  markdown
                                                    st.markdown(markdown_text, unsafe_allow_html=True)

                                                        
                                            elif name_type == 'bachbien':
                                                with st.expander("BẠCH BIẾN"):

                                                    with open('src/title/BACHBIEN.md', 'r', encoding='utf-8') as file:
                                                        markdown_text = file.read()

                                                        # Hiển thị nội dung  markdown
                                                    st.markdown(markdown_text, unsafe_allow_html=True)

                                            des.add(name_type)

                                    if not des:
                                        st.info("Không có dữ liệu để mô tả!")
                        cur_frame += 1
                            
                else:
                    st.error('Không có dữ liệu. Vui lòng chọn một hình ảnh hoặc video về bệnh da liễu!')
    elif select_page == "LIÊN HỆ":
        col1, col2 = st.columns([8.1, 4])
        file_img, file_vid, key_path,vip_cap = '', '', '',''

        with col2:
            st.image(Image.open('src/LOGOBIG.jpg'), width = 200)

            with st.expander("ĐỂ LIÊN HỆ VỚI ADD VUI LÒNG ĐIỀN FORM BÊN TRÁI", expanded = True):
                 with open('src/title/LIENHE.md', 'r', encoding='utf-8') as file:
                     markdown_text = file.read()

                  # Hiển thị nội dung markdown
            st.markdown(markdown_text, unsafe_allow_html=True)

        with col1:
            with open('src/title/INFO1.md', 'r', encoding='utf-8') as file:
                 markdown_text = file.read()

                  # Hiển thị nội dung markdown
            st.markdown(markdown_text, unsafe_allow_html=True)

            name = st.text_input('Họ và tên')
            mail = st.text_input('Email')
            tieude = st.text_input('Tiêu đề')
            noidung = st.text_input('Nội dung')

            recipe_button = st.button('Gửi')

            st.markdown("<hr />", unsafe_allow_html = True)

            if recipe_button:
                if name and mail and tieude and noidung:
                    email_content = f"Họ và tên: {name}\nEmail: {mail}\nTiêu đề: {tieude}\nNội dung: {noidung}"
        
                    # Lưu thông tin vào tệp tin trong thư mục "mess" với mã hóa utf-8
                    file_path = os.path.join("mess", f"{name}_{mail}.txt")
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(email_content)
                    st.success("Phản hồi của bạn đã được gửi thành công!")
                else:
                    st.error('Vui lòng điền đủ thông tin')

    else:
        logo = Image.open('src/GT.jpg')
        st.image(logo, use_column_width = True)
        with open('src/title/GT.md', 'r', encoding='utf-8') as file:
            markdown_text = file.read()

                  # Hiển thị nội dung markdown
        st.markdown(markdown_text, unsafe_allow_html=True)


if __name__ == '__main__':
    main()


import os
import base64
import re
import requests
import streamlit as st
from pathlib import Path
from openai import OpenAI


def replace_image_tag(match):
    image_path = match.group(1)
    file_path = f"{image_path}.desc"

    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            content = file.read()
            return f"此处是文本插图：{content}"

    return match.group(1)


def replace_classify(markdown_text: str):
    markdown_pattern = r'!\[.*?\]\((.*?)\)'
    html_pattern = r'<img .*?src="(.*?)".*?>'
    markdown_text = re.sub(markdown_pattern, replace_image_tag, markdown_text)
    markdown_text = re.sub(html_pattern, replace_image_tag, markdown_text)
    return markdown_text


def deal_md(extract_dir, file_name):
    file_path = f"{extract_dir}{file_name}"
    with open(file_path, 'r') as file:
        md_content = file.read()

        with st.expander(f"{file_path} Original"):
            st.text(md_content)

        updated_md_content = extract_images_from_md(md_content, extract_dir)
        updated_md_content = replace_classify(updated_md_content)

        new_file = f"{file_path}.txt"
        with open(new_file, 'w', encoding='utf-8') as md_file:
            md_file.write(updated_md_content)
            with st.expander(f"{new_file} New"):
                st.text(updated_md_content)


def get_image_description(client, encoded_string, image_extension, prompt, model_choice):
    response = client.chat.completions.create(
        model=model_choice,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/{image_extension};base64,{encoded_string}"}
                    },
                ],
            }
        ],
        max_tokens=300,
    )

    return response.choices[0].message.content


def rek_image(image_path: str):
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    image_classifying_path = f"{image_path}.desc"

    # if image_classifying_path exists, then open it content as string
    if os.path.exists(image_classifying_path):
        with open(image_classifying_path, 'r') as image_file:
            return image_file.read()

    if not os.path.exists(image_path):
        st.write(f"Image not found: {image_path}")
        return ""

    image_extension = image_path.split('.')[-1].split('?')[0]
    with open(image_path, 'rb') as image_file:
        with st.spinner(f'Classifying {image_path} ...'):
            image_data = image_file.read()

            encoded_string = base64.b64encode(image_data).decode('utf-8')
            if encoded_string:
                try:
                    description = get_image_description(
                        client,
                        encoded_string,
                        image_extension,
                        'What’s in this image? please use chinese.',
                        "gpt-4o")

                    with open(image_classifying_path, 'w') as t_file:
                        t_file.write(description)

                    return description
                except Exception as e:
                    st.error(f"Error: {e}")

    return ""


def extract_images_from_md(md_content, extract_dir):
    markdown_image_pattern = r'!\[.*?\]\((.*?)\)'
    html_image_pattern = r'<img\s+.*?src=["\'](.*?)["\']'

    markdown_matches = re.findall(markdown_image_pattern, md_content)
    html_matches = re.findall(html_image_pattern, md_content)

    all_matches = markdown_matches + html_matches

    Path(extract_dir).mkdir(parents=True, exist_ok=True)

    updated_md_content = md_content
    for match in all_matches:
        if match.startswith('data:image'):
            base64_pattern = r'data:image/(.*?);base64,(.*)'
            image_format, base64_data = re.findall(base64_pattern, match)[0]
            image_data = base64.b64decode(base64_data)
            image_filename = f'image_{all_matches.index(match) + 1}.{image_format}'
            image_path = os.path.join(extract_dir, image_filename)
            with open(image_path, 'wb') as image_file:
                image_file.write(image_data)
                rek_image(image_path)
                updated_md_content = updated_md_content.replace(match, image_path)
        elif match.startswith('http') or match.startswith('https'):
            image_path = download_image(match, extract_dir, all_matches.index(match) + 1)
            if image_path:
                rek_image(image_path)
                updated_md_content = updated_md_content.replace(match, image_path)
        else:
            image_path = match
            if not match.startswith(extract_dir):
                image_path = f"{extract_dir}{match}"
            rek_image(image_path)
            updated_md_content = updated_md_content.replace(match, image_path)

    return updated_md_content


def download_image(image_url, output_dir, image_index):
    image_extension = image_url.split('.')[-1].split('?')[0]

    image_extension = image_extension.replace(',', '')
    image_extension = image_extension.replace('&', '')
    image_extension = image_extension.replace('=', '')
    image_extension = image_extension.replace('.', '')

    image_filename = f'image_{image_index}.{image_extension}'
    image_path = os.path.join(output_dir, image_filename)

    try:
        response = requests.get(image_url)
        response.raise_for_status()
        with open(image_path, 'wb') as image_file:
            image_file.write(response.content)
        return image_path
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {image_url}: {e}")
        return None

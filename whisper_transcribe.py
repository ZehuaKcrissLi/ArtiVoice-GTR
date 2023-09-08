from gradio_client import Client
import os
import pinyin
import shutil

client = Client("https://sanchit-gandhi-whisper-jax.hf.space/")

try:
    # 创建或打开一个txt文件来存储结果
    with open('gtr_test.txt', 'w', encoding='utf-8') as f:
        print("File 'gtr_test.txt' is open.")

        counter = 1  # 用于生成新文件名的计数器

        # 遍历指定文件夹中的所有.mp3文件
        for filename in os.listdir("/Users/kcriss/Desktop/gtr_sliced"):
            if filename.endswith(".wav"):
                filepath = os.path.join("/Users/kcriss/Desktop/gtr_sliced", filename)

                # # 生成新文件名并进行重命名
                # new_filename = f"yubo_audio{counter}.wav"
                # new_filepath = os.path.join("/Users/kcriss/Desktop/fanjunrui/voice-converted", new_filename)
                # if not os.path.exists(new_filepath):
                #     shutil.move(filepath, new_filepath)
                #     print(f"Moved {filename} to {new_filename}")
                # else:
                #     print(f"File {new_filename} already exists, skipping.")

                # 调用预测API进行语音转写
                result = client.predict(
                    filepath,
                    "transcribe",
                    False,
                    api_name="/predict_1"
                )
                print(f"API result: {result}")

                # 从元组中提取转写文本
                transcription_text = result[0]

                # 转为拼音
                pinyin_result = pinyin.get(transcription_text, format="strip", delimiter=" ")
                print(f"Pinyin result: {pinyin_result}")

                # 将结果以指定格式写入txt文件
                f.write(f"{filename}|{pinyin_result}\n")
                f.flush()  # 显式刷新缓冲区
                print(f"Written to file: {filename}|{pinyin_result}")

                # 更新计数器
                counter += 1 
except Exception as e:
    print(f"An error occurred: {e}")

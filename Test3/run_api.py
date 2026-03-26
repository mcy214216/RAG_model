from openai import OpenAI
#加载本地的大模型服务
api_key = 'ollama'
base_url = 'http://localhost:11434/v1'
client = OpenAI(api_key=api_key, base_url=base_url)
# # 发送请求到大模型, 流式输出
# response = client.chat.completions.create(model='gemma3:4b', # 使用的模型，可以自行选择
#     messages=[{"role": "system", "content": "You are a helpful assistant."},
#     {"role": "user", "content": "你好，你是谁"},],
#     max_tokens=150, # 返回文本的最大长度
#     temperature=0.7, # 控制生成文本的随机性，值越低，输出越确定
#     stream=True)
#
# # 逐块打印返回结果
# for chunk in response:
#     print(chunk.choices[0].delta.content)
# 打印返回结果
# print(response.choices.delta.content)




# # 发送请求到大模型, 非流式输出
response = client.chat.completions.create(
model='gemma3:4b', # 使用的模型，可以自行选择
messages=[
{"role": "system", "content": "You are a helpful assistant."},
{"role": "user", "content": "你好,你是谁"},
],
max_tokens=150, # 返回文本的最大长度
temperature=0.7, # 控制生成文本的随机性，值越低，输出越确定
stream=False
)
print(response.choices[0].message.content)
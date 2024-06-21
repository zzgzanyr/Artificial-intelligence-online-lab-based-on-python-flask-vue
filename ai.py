from openai import OpenAI

client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = "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"
)

# 修改这里的问题
completion = client.chat.completions.create(
  model="meta/llama3-70b-instruct",
  messages=[{"role":"user","content":"世界上有几个国家？请用中文回复，谢谢你"}],  # 修改了提问内容
  temperature=0.5,
  top_p=1,
  max_tokens=1024,
  stream=True
)

for chunk in completion:
  if chunk.choices[0].delta.content is not None:
    print(chunk.choices[0].delta.content, end="")

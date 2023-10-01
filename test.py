from profanity_check import predict

text = ["Hello, this is a clean text.", "fuck you!"]

results = predict(text)

for i, result in enumerate(results):
    if result == 1:
        print(f"Text {i+1}: 욕설이 포함되어 있습니다.")
    else:
        print(f"Text {i+1}: 욕설이 없습니다.")
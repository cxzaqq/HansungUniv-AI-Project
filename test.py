from profanity_check import predict, predict_prob

text = ["Hello, this is a clean text.", "fukc you!"]

results = predict_prob(text)

for i, result in enumerate(results):
    print(results)


# for i, result in enumerate(results):
#     if result == 1:
#         print(f"Text {i+1}: 욕설이 포함되어 있습니다.")
#     else:
#         print(f"Text {i+1}: 욕설이 없습니다.")
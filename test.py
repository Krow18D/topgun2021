categoryList = {
        "lime":{"count":0,"size":[]},
        "marker":{"count":0,"size":[1,2,3]}
    }

x = categoryList["lime"]["count"] = categoryList["lime"]["count"] + 1
print(x)
categoryList["lime"]["size"].append(1)
print(categoryList["lime"]["size"][categoryList["lime"]["count"]-1])
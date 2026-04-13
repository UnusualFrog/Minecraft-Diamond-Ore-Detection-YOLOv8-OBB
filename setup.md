# Steps for Running Model

## Step 1 - Download dependencies
```
pip install --upgrade pip
pip install -r requirements.txt
```

## Step 2 - Train Model
- Run final.py to train
- Select option 1 from the terminal menu to train the model to the dataset
- Outputs will be saved to `.\runs\obb\diamond_ore`
- Subsequent outputs will be saved to `.\runs\obb\diamond_oreX` where X is the number of times the model has been trained

## Step 3 - Validate Model
- Select option 2 to validate the trained model
- Outputs will be saved to `.\runs\obb\validate`
- Subsequent outputs will be saved to `.\runs\obb\validateX` where X is the number of times the model has been validated

## Step 4 - Unseen data
- Select option 3 to perform predictions on unseen data
- For unseen data, ensure Minecraft screenshots are  either .png or .jpg filetypes and are placed in the directory `.\Data\predict\`
- Ex. `.\Data\predict\screenshot_diamonds_01.png`
- Outputs will be saved to `.\runs\obb\predict`
- Subsequent outputs will be saved to `.\runs\obb\predictX` where X is the number of times the model has predicted on unseen data




import webcolors
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from colormap import rgb2hex
import pandas as pd
from scipy.spatial import KDTree


class ColorNames:

    WebColorMap = {}
    WebColorMap["AliceBlue"] = "#F0F8FF"
    WebColorMap["AntiqueWhite"] = "#FAEBD7"
    WebColorMap["Aqua"] = "#00FFFF"
    WebColorMap["Aquamarine"] = "#7FFFD4"
    WebColorMap["Azure"] = "#F0FFFF"
    WebColorMap["Beige"] = "#F5F5DC"
    WebColorMap["Bisque"] = "#FFE4C4"
    WebColorMap["Black"] = "#000000"
    WebColorMap["BlanchedAlmond"] = "#FFEBCD"
    WebColorMap["Blue"] = "#0000FF"
    WebColorMap["BlueViolet"] = "#8A2BE2"
    WebColorMap["Brown"] = "#A52A2A"
    WebColorMap["BurlyWood"] = "#DEB887"
    WebColorMap["CadetBlue"] = "#5F9EA0"
    WebColorMap["Chartreuse"] = "#7FFF00"
    WebColorMap["Chocolate"] = "#D2691E"
    WebColorMap["Coral"] = "#FF7F50"
    WebColorMap["CornflowerBlue"] = "#6495ED"
    WebColorMap["Cornsilk"] = "#FFF8DC"
    WebColorMap["Crimson"] = "#DC143C"
    WebColorMap["Cyan"] = "#00FFFF"
    WebColorMap["DarkBlue"] = "#00008B"
    WebColorMap["DarkCyan"] = "#008B8B"
    WebColorMap["DarkGoldenRod"] = "#B8860B"
    WebColorMap["DarkGray"] = "#A9A9A9"
    WebColorMap["DarkGrey"] = "#A9A9A9"
    WebColorMap["DarkGreen"] = "#006400"
    WebColorMap["DarkKhaki"] = "#BDB76B"
    WebColorMap["DarkMagenta"] = "#8B008B"
    WebColorMap["DarkOliveGreen"] = "#556B2F"
    WebColorMap["Darkorange"] = "#FF8C00"
    WebColorMap["DarkOrchid"] = "#9932CC"
    WebColorMap["DarkRed"] = "#8B0000"
    WebColorMap["DarkSalmon"] = "#E9967A"
    WebColorMap["DarkSeaGreen"] = "#8FBC8F"
    WebColorMap["DarkSlateBlue"] = "#483D8B"
    WebColorMap["DarkSlateGray"] = "#2F4F4F"
    WebColorMap["DarkSlateGrey"] = "#2F4F4F"
    WebColorMap["DarkTurquoise"] = "#00CED1"
    WebColorMap["DarkViolet"] = "#9400D3"
    WebColorMap["DeepPink"] = "#FF1493"
    WebColorMap["DeepSkyBlue"] = "#00BFFF"
    WebColorMap["DimGray"] = "#696969"
    WebColorMap["DimGrey"] = "#696969"
    WebColorMap["DodgerBlue"] = "#1E90FF"
    WebColorMap["FireBrick"] = "#B22222"
    WebColorMap["FloralWhite"] = "#FFFAF0"
    WebColorMap["ForestGreen"] = "#228B22"
    WebColorMap["Fuchsia"] = "#FF00FF"
    WebColorMap["Gainsboro"] = "#DCDCDC"
    WebColorMap["GhostWhite"] = "#F8F8FF"
    WebColorMap["Gold"] = "#FFD700"
    WebColorMap["GoldenRod"] = "#DAA520"
    WebColorMap["Gray"] = "#808080"
    WebColorMap["Grey"] = "#808080"
    WebColorMap["Green"] = "#008000"
    WebColorMap["GreenYellow"] = "#ADFF2F"
    WebColorMap["HoneyDew"] = "#F0FFF0"
    WebColorMap["HotPink"] = "#FF69B4"
    WebColorMap["IndianRed"] = "#CD5C5C"
    WebColorMap["Indigo"] = "#4B0082"
    WebColorMap["Ivory"] = "#FFFFF0"
    WebColorMap["Khaki"] = "#F0E68C"
    WebColorMap["Lavender"] = "#E6E6FA"
    WebColorMap["LavenderBlush"] = "#FFF0F5"
    WebColorMap["LawnGreen"] = "#7CFC00"
    WebColorMap["LemonChiffon"] = "#FFFACD"
    WebColorMap["LightBlue"] = "#ADD8E6"
    WebColorMap["LightCoral"] = "#F08080"
    WebColorMap["LightCyan"] = "#E0FFFF"
    WebColorMap["LightGoldenRodYellow"] = "#FAFAD2"
    WebColorMap["LightGray"] = "#D3D3D3"
    WebColorMap["LightGrey"] = "#D3D3D3"
    WebColorMap["LightGreen"] = "#90EE90"
    WebColorMap["LightPink"] = "#FFB6C1"
    WebColorMap["LightSalmon"] = "#FFA07A"
    WebColorMap["LightSeaGreen"] = "#20B2AA"
    WebColorMap["LightSkyBlue"] = "#87CEFA"
    WebColorMap["LightSlateGray"] = "#778899"
    WebColorMap["LightSlateGrey"] = "#778899"
    WebColorMap["LightSteelBlue"] = "#B0C4DE"
    WebColorMap["LightYellow"] = "#FFFFE0"
    WebColorMap["Lime"] = "#00FF00"
    WebColorMap["LimeGreen"] = "#32CD32"
    WebColorMap["Linen"] = "#FAF0E6"
    WebColorMap["Magenta"] = "#FF00FF"
    WebColorMap["Maroon"] = "#800000"
    WebColorMap["MediumAquaMarine"] = "#66CDAA"
    WebColorMap["MediumBlue"] = "#0000CD"
    WebColorMap["MediumOrchid"] = "#BA55D3"
    WebColorMap["MediumPurple"] = "#9370D8"
    WebColorMap["MediumSeaGreen"] = "#3CB371"
    WebColorMap["MediumSlateBlue"] = "#7B68EE"
    WebColorMap["MediumSpringGreen"] = "#00FA9A"
    WebColorMap["MediumTurquoise"] = "#48D1CC"
    WebColorMap["MediumVioletRed"] = "#C71585"
    WebColorMap["MidnightBlue"] = "#191970"
    WebColorMap["MintCream"] = "#F5FFFA"
    WebColorMap["MistyRose"] = "#FFE4E1"
    WebColorMap["Moccasin"] = "#FFE4B5"
    WebColorMap["NavajoWhite"] = "#FFDEAD"
    WebColorMap["Navy"] = "#000080"
    WebColorMap["OldLace"] = "#FDF5E6"
    WebColorMap["Olive"] = "#808000"
    WebColorMap["OliveDrab"] = "#6B8E23"
    WebColorMap["Orange"] = "#FFA500"
    WebColorMap["OrangeRed"] = "#FF4500"
    WebColorMap["Orchid"] = "#DA70D6"
    WebColorMap["PaleGoldenRod"] = "#EEE8AA"
    WebColorMap["PaleGreen"] = "#98FB98"
    WebColorMap["PaleTurquoise"] = "#AFEEEE"
    WebColorMap["PaleVioletRed"] = "#D87093"
    WebColorMap["PapayaWhip"] = "#FFEFD5"
    WebColorMap["PeachPuff"] = "#FFDAB9"
    WebColorMap["Peru"] = "#CD853F"
    WebColorMap["Pink"] = "#FFC0CB"
    WebColorMap["Plum"] = "#DDA0DD"
    WebColorMap["PowderBlue"] = "#B0E0E6"
    WebColorMap["Purple"] = "#800080"
    WebColorMap["Red"] = "#FF0000"
    WebColorMap["RosyBrown"] = "#BC8F8F"
    WebColorMap["RoyalBlue"] = "#4169E1"
    WebColorMap["SaddleBrown"] = "#8B4513"
    WebColorMap["Salmon"] = "#FA8072"
    WebColorMap["SandyBrown"] = "#F4A460"
    WebColorMap["SeaGreen"] = "#2E8B57"
    WebColorMap["SeaShell"] = "#FFF5EE"
    WebColorMap["Sienna"] = "#A0522D"
    WebColorMap["Silver"] = "#C0C0C0"
    WebColorMap["SkyBlue"] = "#87CEEB"
    WebColorMap["SlateBlue"] = "#6A5ACD"
    WebColorMap["SlateGray"] = "#708090"
    WebColorMap["SlateGrey"] = "#708090"
    WebColorMap["Snow"] = "#FFFAFA"
    WebColorMap["SpringGreen"] = "#00FF7F"
    WebColorMap["SteelBlue"] = "#4682B4"
    WebColorMap["Tan"] = "#D2B48C"
    WebColorMap["Teal"] = "#008080"
    WebColorMap["Thistle"] = "#D8BFD8"
    WebColorMap["Tomato"] = "#FF6347"
    WebColorMap["Turquoise"] = "#40E0D0"
    WebColorMap["Violet"] = "#EE82EE"
    WebColorMap["Wheat"] = "#F5DEB3"
    WebColorMap["White"] = "#FFFFFF"
    WebColorMap["WhiteSmoke"] = "#F5F5F5"
    WebColorMap["Yellow"] = "#FFFF00"
    WebColorMap["YellowGreen"] = "#9ACD32"

    @staticmethod
    def rgbFromStr(s):
        # s starts with a #.
        r, g, b = int(s[1:3], 16), int(s[3:5], 16), int(s[5:7], 16)
        return r, g, b

    @staticmethod
    def findNearestWebColorName(R,G,B):
        return ColorNames.findNearestColorName(R,G,B, ColorNames.WebColorMap)


    @staticmethod
    def findNearestColorName(R,G,B, Map):
        mindiff = None
        for d in Map:
            r, g, b = ColorNames.rgbFromStr(Map[d])
            diff = abs(R - r) * 256 + abs(G - g) * 256 + abs(B - b) * 256
            if mindiff is None or diff < mindiff:
                mindiff = diff
                mincolorname = d
        return mincolorname



def calculate_white(img):
    clusters = 3
    dc = DominantColors(img, clusters)
    colors = dc.dominantColors()
    percentage = dc.get_percentage()
    r = img.split('/')
    category =  r[1]
    name = r[2]
    col = ""
    max = 0
    maxColor = ""
    for i in range(len(colors)):
        hex = str(rgb2hex(colors[i][0], colors[i][1], colors[i][2]))
        col = col + hex + '(' + str(percentage[i]) + '),'
        if max < percentage[i]:
            max = percentage[i]
            maxColor = colors[i]

    col = col [:-1]
    maxColor = ColorNames.findNearestWebColorName(maxColor[0],maxColor[1],maxColor[2])

    dict = {
        "category": category,
        "name": name,
        "color": col,
        "mainColor": maxColor

    }
    print(dict)
    return dict


class DominantColors:
    CLUSTERS = None
    IMAGE = None
    COLORS = None
    LABELS = None

    def __init__(self, image, clusters=3):
        self.CLUSTERS = clusters
        self.IMAGE = image

    def dominantColors(self):
        # read image
        from PIL import Image

        im = Image.open(self.IMAGE, 'r')
        pixel_values = list(im.getdata())
        pixels = []
        for pv in pixel_values:
            if (pv[3] > 0):
                pixels.append(pv[:-1])


        if len(pixels) == 0:
            pixels.append([0,0,0])

        img = self.IMAGE
        # save image after operations
        self.IMAGE = pixels

        # using k-means to cluster pixels
        diff = 0

        done = False


        if len(pixels) < self.CLUSTERS:
            self.IMAGE = []
            for p in pixels:
                for r in range(self.CLUSTERS * 10):
                    self.IMAGE.append(p)

        while not done:
            try:
                kmeans = KMeans(n_clusters=self.CLUSTERS - diff)
                kmeans.fit(self.IMAGE)
                done = True
            except ValueError:
                print("------------------------ERROR---------------------------------------" + str(img))
                diff = diff + 1
                if diff > self.CLUSTERS:
                    break

        # the cluster centers are our dominant colors.
        self.COLORS = kmeans.cluster_centers_

        # save labels
        self.LABELS = kmeans.labels_

        # returning after converting to integer from float
        return self.COLORS.astype(int)

    def get_percentage(self):
        from collections import Counter, defaultdict
        total = 0
        counter = {}
        c = Counter(self.LABELS)
        for key in sorted(c):
            counter[key] = c[key]

        for k, v in counter.items():
            total = total + v
        percentage = {}
        for k, v in counter.items():
            percentage[k] = v / total * 100

        return percentage



import os

images = []
for root, dirs, files in os.walk("test2"):
    for dir in dirs:
        for root, dirs, files in os.walk("test2/" + dir):
            for file in files:
                images.append("test2/" + dir + "/" + file)





n = len(images)


from multiprocessing.dummy import Pool as ThreadPool
pool = ThreadPool(16)
results = pool.map(calculate_white, images)

results = pd.DataFrame(results)

print(results)

results.to_csv('colors.csv', encoding='utf-8', index=False)




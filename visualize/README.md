# Data Visualization
파이썬에서 주로 활용되는 시각화 라이브러리 matplotlib를 이용하여 데이터 시각화를 함

# Install
```
pip install matplotlib
```

# anatomy of figure
그림의 각 명칭도  
![anatomy](fig/sphx_glr_anatomy_001.png)

# draw plot
plot 그리기  
![plot](graph/plot.png")
```
import matplotlib.pyplot as plt

plt.plot([1,2,3,4], [2,4,6,8])
plt.plot([1,2,3,4], [1,2,3,4])
plt.show()
```
plot 함수에 두 개의 리스트를 입력하면 각 x축, y축 값으로 지정됨
plot 함수를 여러개 사용한다면 여러개의 plot을 그릴 수 있음

## xlabel, ylabel
```
plt.xlabel('X-Label')
plt.ylabel('Y-Label')
```
x축과 y축의 이름을 지정해 줄 수 있음

## color&style
![color&style](graph/color&style.png")
선의 색깔 및 스타일 변경  
```
plt.plot([1,2,3,4], [2,4,6,8], 'r--')
plt.plot([1,2,3,4], [1,2,3,4], 'm-.')
```
자세한 사항은 아래 그림 참조
![line_color](graph/line_color.png")
![line_style](graph/line_style.png")

## grid
격자를 그리기  
![grid](graph/grid.png")
```
plt.plot([1,2,3,4], [2,4,6,8])
plt.plot([1,2,3,4], [1,2,3,4])
plt.grid(True)
```

## hlines&vlines
그래프 내에 수직 및 수평선을 추가  
![lines](graph/lines.png")
```
plt.plot([1,2,3,4], [2,4,6,8])
plt.plot([1,2,3,4], [1,2,3,4])
plt.axhline(y=3, color='r', linestyle='--',  linewidth=1)
plt.axvline(x=2, color='b', linestyle='-.',  linewidth=1)
```

# draw bar graph
막대 그래프 그리기
![bar](graph/bar.png")
```
x = [0,1,2]
years = ['2020', '2021', '2022']
prices = [500, 450, 700]

plt.bar(x, prices)
plt.xticks(x, years)
```

# draw barh graph
수평 막대 그래프 그리기
![barh](graph/barh.png")
```
x = [0,1,2]
years = ['2020', '2021', '2022']
prices = [500, 450, 700]

plt.barh(x, prices, tick_label=years)
```

# draw scatter plot
산점도 그리기  
![scatter](graph/scatter.png")
```
import numpy as np
N = 50
x = np.random.rand(N)
y = np.random.rand(N)
colors = np.random.rand(N)
size = (100 * np.random.rand(N))
plt.scatter(x, y, s=size, c=colors, alpha=0.5)
```
s는 각 산점도의 크기
c는 각 산점도의 색깔
alpha는 산점도의 투명도를 나타냄

# draw historgram
히스토그램 그리기  
![histogram](graph/histogram.png")
```
height = [170, 165, 174, 176, 173, 167, 185, 184, 183, 174, 191, 164, 153, 153, 175, 175, 178, 156, 146, 185, 175, 177]
plt.hist(height)
```

# draw pie chart
원그래프 그리기  
![pie](graph/pie.png")
```
ratio = [27, 33, 15, 25]
labels = ['Bear', 'Soju', 'Wine', 'Makuly']
plt.pie(ratio, labels=labels, autopct='%.2f%%')
```
autopc는 숫자 표현 형식으로 소숫점 아래 두 자리 까지 표현함
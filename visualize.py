# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 14:41:14 2018

@author: xingxf03
"""

import visdom
import numpy as np
#import matplotlib.pyplot as plt
vis = visdom.Visdom()
#textwindow = vis.text('Hello, world!')
#updatetextwindow = vis.text('Hello World! More text should be here')
#vis.text('And here it is', win=updatetextwindow, append=True)

#vis.boxplot([1,23,2,4])


#vis.video(tensor=video)
   
#image Demo
#vis.image(
#        np.random.randn(3,512,256),
#        opts=dict(title='Random!',caption='How Random'),
#            )

#vis.images(
#        np.random.randn(20,3,64,64),
#        opts=dict(title='Random iamges',caption='How Random'),
#            )

### scatter plots
#Y = np.random.rand(100)
#old_scatter=vis.scatter(
#        X = np.random.rand(100,2),
#        Y = (Y[Y>0]+1.5).astype(int),
#        opts = dict(
#                legend=['Didnt','Update'],
#                xtickmin=-50,
#                xtickmax=50,
#                xtickstep=0.5,
#                ytickmin=-50,
#                ytickmax=50,
#                ytickstep=0.5,
#                markersymbol='cross-thin-open',   
#                )       
#        )
#vis.update_window_opts(
#        win=old_scatter,
#        opts=dict(
#            legend=['Apples', 'Pears'],
#            xtickmin=0,
#            xtickmax=1,
#            xtickstep=0.5,
#            ytickmin=0,
#            ytickmax=1,
#            ytickstep=0.5,
#            markersymbol='cross-thin-open',
#        ),
#    )
#vis.scatter(
#        X=np.random.rand(100, 3),
#        Y=(Y + 1.5).astype(int),
#        opts=dict(
#            legend=['Men', 'Women'],
#            markersize=5,
#        )
#    )
#        # 2D scatterplot with custom intensities (red channel)
#vis.scatter(
#        X=np.random.rand(255, 2),
#        Y=(np.random.rand(255) + 1.5).astype(int),
#        opts=dict(
#            markersize=10,
#            markercolor=np.random.randint(0, 255, (2, 3,)),
#        ),
#    )
#win= vis.scatter(
#        X=np.random.rand(255, 2),
#        opts=dict(
#            markersize=10,
#            markercolor=np.random.randint(0, 255, (255, 3,)),
#        ),
#    )
#        
## add new trace to scatter plot
#vis.scatter(
#        X=np.random.rand(255),
#        Y=np.random.rand(255),
#        win=win,
#        name='new_trace',
#        update='new'
#    )
# bar plots
#vis.bar(X=np.random.rand(20))
#vis.bar(
#        X =  np.abs(np.random.rand(5,3)),
#        opts = dict(
#                stacked = True,
#                legend=['Facebook','Google','Twitter'],
#                rownames=['2012','2013','2014','2015','2016']
#                
#                )
#        
#        )

####histogram
#vis.histogram(X=np.random.rand(10000),opts=dict(numbins=20))

# heatmap
#vis.heatmap(
#        X=np.outer(np.arange(1, 6), np.arange(1, 11)),
#        opts=dict(
#            columnnames=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'],
#            rownames=['y1', 'y2', 'y3', 'y4', 'y5'],
#            colormap='Electric',
#        )
#    )

 #### contour 轮廓线
#x = np.tile(np.arange(1, 101), (100, 1))
#y = x.transpose()
#X = np.exp((((x - 50) ** 2) + ((y - 50) ** 2)) / -(20.0 ** 2))
#vis.contour(X=X, opts=dict(colormap='Viridis'))

#####surface  表面图
#vis.surf(X=X,opts=dict(colormap='Hot'))

#line plots
#vis.line(Y=np.random.rand(10),opts=dict(showlegend=True))
#Y = np.linspace(-5, 5, 100)
#vis.line(
#        Y=np.column_stack((Y * Y, np.sqrt(Y + 5))),
#        X=np.column_stack((Y, Y)),
#        opts=dict(markers=False),
#    )


# boxplot
X = np.random.rand(100, 2)
X[:, 1] += 2
vis.boxplot(
        X=X,
        opts=dict(legend=['Men', 'Women'])
    )














































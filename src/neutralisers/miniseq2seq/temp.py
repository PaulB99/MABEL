# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 20:19:01 2021

@author: paulb
"""
import os
file = open('bert-base-uncased-vocab.txt', 'r', encoding='utf-8')
lines = file.readlines()
counter = 0
newlines = []
for line in lines:
    if counter<7630:
        newlines.append(lines)
        counter+=1
        
print(type(newlines[0][0]))
os.remove('bert-base-uncased-vocab-mini.txt')
file2 = open('bert-base-uncased-vocab-mini.txt', 'w+', encoding='utf-8')
file2.writelines(newlines[0])

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('employee_data.csv')
# print(df.head())
# print('--------------[1]---------------')
# print(df.info())
# print('--------------[2]---------------')
# print(df.describe())


# 특정 열 선택
names = df['이름']
# print(names.head())

first_row = df.iloc[0]
# print(first_row)

older_than_30 = df[df['나이'] > 30]

# print(df['나이'] > 30)
# print(older_than_30.head())

grouped_df = df.groupby('부서')['나이'].mean()
#print(grouped_df)

df['연령대'] = df['나이'].apply(lambda x: '30대' if 30 <= x < 40 else '30대 이하' if x < 30 else '40대 이상')

df.to_csv('/mnt/data/modified_employee_data.csv', index=False)
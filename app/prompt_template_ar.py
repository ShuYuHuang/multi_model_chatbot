######
# Prompt Template list for different functions
######
__all__ = [
    'QUERY_INTEGRATION_TEMPLATE',
    'DATA_SHORT_DESCRIPTION',
    'INITIAL_CSV_PLOT',
    'PRMPTED_CSV_PLOT',
    'DISCRIBE_PLOT'
]
# ------------------------------------------------------------------------------------------
QUERY_INTEGRATION_TEMPLATE  = """
طلب:

ترتبط جميع الأسئلة إما بنتائج الاستعلام أو التاريخ،
يرجى الرد على الأسئلة المطروحة فقط، وعدم التوسيع إلى أسئلة أخرى

إذا كان السؤال متعلقًا بالاستعلام عن النتائج:
- استخراج المعلومات من النتيجة التي تم الاستعلام عنها والإجابة على السؤال
وإلا إذا كان السؤال متعلقًا بالتاريخ:
- استخراج المعلومات من التاريخ والإجابة على السؤال

قاعدة بيانات نموذج النتائج الاستعلام عنها:
```
{queried}
```

تاريخ:
```
{history}
```

مطلوب منك:
{input}

يرد:
"""

# ------------------------------------------------------------------------------------------
DATA_SHORT_DESCRIPTION  = """
طلب:
وفيما يلي جدول البيانات الذي يستخدم للمساعدة في شرح بعض الاتجاهات.

بيانات الجدول:
```
{table_data}
```

إعطاء وصف مختصر للبيانات.
يرد:
"""
# ------------------------------------------------------------------------------------------
# PRMPTED_CSV_PLOT = """
# الطلب:
# أنت مبرمج مثالي في لغة البرمجة Python.
# هذا جزء من ملف CSV الخاص بي بالاسم {filename}:

# ```
# {head3lines}
# ```
# تم حفظه كـ pandas DataFrame بمتغير يسمى csv_df.

# التعليمات:
# ```
# {instructions}
# ```

# يرجى استخدام csv_df وإنشاء الشيفرة <code> لرسم csv_df في plotly بالتنسيق المطلوب.
# لا تستخدم pivot_table.
# يجب استخدام plotly.graph_objs كـ go.
# انتبه إلى المسافات في اسم الحقل في الرسم البياني.
# يجب أن تكون الحلاقة باستخدام plotly و plotly فقط.
# لا تستخدم matplotlib.
# يرجى إرجاع الشيفرة <code> في الشكل التالي ```python <code>```
# """

PRMPTED_CSV_PLOT = """
You are a perfect data scientist masters python.

This is the part of my CSV file:
```
,date,day,aircraft,helicopter,tank,APC,field artillery,MRL,military auto,fuel tank,drone,naval ship,anti-aircraft warfare,special equipment,mobile SRBM system,greatest losses direction,vehicles and fuel tanks,cruise missiles,submarines
0,2022-02-25,2,10,7,80,516,49,4,100.0,60.0,0,2,0,,,,,,
1,2022-02-26,3,27,26,146,706,49,4,130.0,60.0,2,2,0,,,,,,
2,2022-02-27,4,27,26,150,706,50,4,130.0,60.0,2,2,0,,,,,,
```
It have been saved as DataFrame csv_df.

Aim of investigation:
```{project_aim}```

Instruction:
```{instructions}```

Generate the harmless simple code CODE deal with csv_df according to the requirements.
Here are some cases that may used to correct the output codes:
CASE 1: If the instructions is for plotting:
    Dont use pd.read_csv, use variable csv_df directly
    Don't use pivot_table.
    Mind the spaces included in the field name of plot.
    The solution should be given using plotly and only plotly.
    Do not use matplotlib.
    result should be stored in variable 'fig'
CASE 2: If the instructions is for text or table generation:
    Instead of using pd.read_csv, use variable csv_df directly
    The result of code should be stored in a markdown string format called 'resp'
    Don't do anything related to 'resp' if it is for plotting

Return the code CODE in the following format:
```python (started from "```python")
CODE
``` (ended by "```")
"""

INITIAL_PLOT_INSTRUCTION = """رسم مخطط خطي للميزات"""

DISCRIBE_PLOT = """
وفقًا للطلب {user_input}، قم بتقديم وصف لـ <image> هذا
الرد باللغة العربية
"""


# Orig
# According to these information, 
# generate the code <code> for plotting the previous data in plotly, in the format requested. 
# The solution should be given using plotly and only plotly. Do not use matplotlib.
# Return the code <code> in the following format ```python <code>```
# """

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
import statsmodels.api as sm
import scipy.stats as stats

# https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python

DATA_PATH = 'Data'


def retrieve_dataframe_from_csv(file_name='iris_data.csv'):
    """

    :param file_name:
        file name of the csv that contains the dataframe
    :return:
        dataframe extracted from the passed path
    """
    return pd.read_csv(os.path.join(DATA_PATH, file_name))


def validity_check_dataframe(df):
    """
        :param df:
            dataframe with the variable to describe
        :return: dataframe
        """
    try:
        if type(df) == pd.core.frame.DataFrame:
            if len(df) > 1:
                return df
            else:
                raise DataframeTooSmall
        else:
            raise DataframeNotDefinedError
    except DataframeNotDefinedError:
        print("No dataframe defined")
    except DataframeTooSmall:
        print("No dataframe defined")


def describe_var(df, var_name):
    """
    :param df:
        dataframe with the variable to describe
    :param var_name:
        the column to describe of the dataframe df
    :return:
    """
    df = validity_check_dataframe(df)
    description = (df[var_name].describe())
    return description


def plot_var(df, var_name):
    """
    :param df:
        dataframe with the variable to describe
    :param var_name:
        the column to describe of the dataframe df
    :return:
    """
    df = validity_check_dataframe(df)
    kurtosis = df[var_name].kurt()
    skewness = df[var_name].skew()
    full_description = " {} \n kurtosis : {}, skewness: {} ".format(describe_var(df, var_name), kurtosis, skewness)
    axes_subplot_seaborn = sns.distplot(df[var_name])
    axes_subplot_seaborn.set(title=var_name)
    # full description (includes kurtosis and skewness)
    print(full_description)
    plt.show()


def scatter_plot_numeric_var(df, var_1, var_2):
    """
    :param df:
        dataframe where we take the data
    :param var_1:
        first variable to take into account in the scatter plto
    :param var_2:
        second variable to take into account in the scatter plto
    :return:
    """
    data = pd.concat([df[var_1], df[var_2]], axis=1)
    data.plot.scatter(x=var_1, y=var_2)
    plt.show()


def anova_test(df, x, y, reference_value='', intercept=True):
    """
    :param df:
    :param x:
        x variable of the regression analysis (categoric)
    :param y:
        y variable of the regression analysis (numeric)
    :param reference_value:
        if the anova test needs one reference value
    :return:
    """
    if reference_value:
        reference_value = ", Treatment(reference='{}')".format(reference_value)
    formula = "{} ~ C({}{})".format(y, x, reference_value)

    if not intercept:
        formula += " - 1"

    print(formula)
    results = ols(formula, data=df).fit()
    print(results.summary())
    sm.stats.anova_lm(results)


def literal_substitution(df, var):
    """
    :param df:
        dataframe
    :param var:
        categorical var of the dataframe where the categorical values will be replaced with numeric values
    :return:
        dataframe with the replaced var
    """
    # substitute literals
    set_dict = set(list(df[var]))
    integer_dict = {}
    for i in range(len(set_dict)):
        elem = set_dict.pop()
        integer_dict.update({elem: i})
    print("The replace will be : {}".format(integer_dict))
    df[var] = df[var].replace(integer_dict)
    return df


def t_test(df, selected_var, group_var, comparison_group=[]):
    if not comparison_group:
        return
    # copy
    df_ttest = df.copy()

    # substitute literals
    df_ttest = literal_substitution(df_ttest, group_var)

    # select subset
    rvs1 = df_ttest[df_ttest[group_var] == 0]
    rvs2 = df_ttest[df_ttest[group_var] == 1]

    # sta_ = stats.ttest_ind(df_ttest[selected_var], df_ttest[independent_var])
    sta_ = stats.ttest_ind(rvs1[selected_var], rvs2[selected_var])
    print(sta_)


def main():
    file_name = 'iris_data.csv'
    df = retrieve_dataframe_from_csv(file_name)
    # sepal_length,sepal_width,petal_length,petal_width,label
    # describe_var(df, 'sepal_length')
    # plot_var(df, 'sepal_length')
    # scatter_plot_numeric_var(df, 'sepal_length', 'sepal_width')
    # anova_test(df, 'label', 'sepal_length', intercept=False)
    # t_test(df, 'petal_length', 'label', comparison_group=['Iris-setosa', 'Iris-versicolor'])


if __name__ == "__main__":
    main()


class CustomError(Exception):
    pass


class DataframeNotDefinedError(CustomError):
    """raised when the dataframe is not defined"""
    pass


class DataframeTooSmall(CustomError):
    """raised when the dataframe is too small"""
    pass

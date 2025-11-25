{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/hrudhay67/ML_LAB_2/blob/main/Candidate_elimination.ipynb\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "t6fsw0Yofiqt",
        "outputId": "a6328b10-be01-4201-de5a-bcd134df8dcc"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Most Specific Hypothesis (S): [['?', '?', '?', '?', '?', '?']]\n",
            "Most General Hypothesis (G): []\n"
          ]
        }
      ],
      "source": [
        "def candidate_elimination_algorithm(dataset, attributes):\n",
        "    num_attributes = len(attributes)\n",
        "    S = None\n",
        "    G = [['?' for _ in range(num_attributes)]]\n",
        "\n",
        "    for example in dataset:\n",
        "        instance = example[:-1]\n",
        "        label = example[-1]\n",
        "\n",
        "        if label == 'Yes':\n",
        "            if S is None:\n",
        "                S = instance.copy()\n",
        "            else:\n",
        "                for i in range(num_attributes):\n",
        "                    if S[i] != instance[i]:\n",
        "                        S[i] = '?'\n",
        "\n",
        "            G = [g for g in G if all(g[i] == '?' or g[i] == instance[i] for i in range(num_attributes))]\n",
        "\n",
        "        else:\n",
        "            G_new = []\n",
        "            for g in G:\n",
        "                for i in range(num_attributes):\n",
        "                    if g[i] == '?':\n",
        "                        if S[i] != '?':\n",
        "                            new_hypothesis = g.copy()\n",
        "                            new_hypothesis[i] = S[i]\n",
        "                            if all(new_hypothesis[j] == '?' or new_hypothesis[j] == instance[j] for j in range(num_attributes)) == False:\n",
        "                                G_new.append(new_hypothesis)\n",
        "            G = G_new\n",
        "\n",
        "    return [S], G\n",
        "\n",
        "dataset = [\n",
        "  ['Senior','low','No','Fair','Unemployed','No','Yes'],\n",
        "  ['Young','High','No','Fair','Employed','Yes','No'],\n",
        "  ['Young','Medium','Yes','Excellent','Employed','No','No'],\n",
        "  ['Middle','High','Yes','Excellent','Employed','Yes','Yes'],\n",
        "  ['Young','Low','Yes','Fair','Unemployed','No','Yes'],\n",
        "  ['Senior','Medium','No','Fair','Employed','Yes','Yes'],\n",
        "]\n",
        "\n",
        "attributes = ['Age','Income','Student','Credit_Rating','Employed','Collateral']\n",
        "\n",
        "S, G = candidate_elimination_algorithm(dataset, attributes)\n",
        "\n",
        "print(\"Most Specific Hypothesis (S):\", S)\n",
        "print(\"Most General Hypothesis (G):\", G)\n"
      ]
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "s-gx9easl3H2"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}
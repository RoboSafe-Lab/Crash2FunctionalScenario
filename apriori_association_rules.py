import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules


def run_apriori(input_file, output_file, min_support, min_confidence, min_lift):
    # Read the xlsx file
    df = pd.read_excel(input_file)

    # Convert the data into the transaction format required by Apriori
    df = df.drop('id', axis=1)
    df = df.astype(str)
    df = df.apply(lambda x: ' '.join(x), axis=1).str.get_dummies(sep=' ')

    # Cast the one-hot encoded frame to boolean
    df = df.astype(bool)

    # Run the Apriori algorithm
    frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

    # Keep only the columns to be exported
    rules = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]

    # Filter the rules by the minimum lift
    rules = rules[rules['lift'] > min_lift]

    # Convert the frozensets to strings so that they can be written out
    rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
    rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))

    # Write the results to an xlsx file
    rules.to_excel(output_file, index=False)

    print(f"Apriori results have been written to: {output_file}")


# Parameters
input_file = ''
output_file = ''
min_support = 0.05
min_confidence = 0.8
min_lift = 1

# Run the Apriori algorithm
run_apriori(input_file, output_file, min_support, min_confidence, min_lift)

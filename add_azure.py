
with open('mkdocs.yml', 'r') as f:
    content = f.read()

# Replace the first occurrence of models/litellm.md with itself + azure
modified = content.replace('                    - models/litellm.md', '                    - models/litellm.md\n                    - models/azure.md', 1)

with open('mkdocs.yml', 'w') as f:
    f.write(modified)

print('Successfully updated mkdocs.yml to include azure.md')
 EOF
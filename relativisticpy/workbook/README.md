Workbook Module:

1. User input (string, file, cell) -> workbook
2. Workbook generates a tree.
3. Workbook applies (tree processor + state) to execute tree.
4. tree processor implements each node individually, sometimes using NodeHandlers (for more complex nodes such as the tensor nodes etc...)
5. Workbook outputs all executed results as a list.
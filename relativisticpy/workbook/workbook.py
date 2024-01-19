import re
import json

from relativisticpy.parsers import RelParser

from relativisticpy.workbook.ast_visitor import RelPyAstNodeTraverser
from relativisticpy.workbook.state import WorkbookState


class Workbook:

    _cache = WorkbookState()
    parser = RelParser(
        RelPyAstNodeTraverser(_cache),
        RelPyAstNodeTraverser.node_configuration,
        RelPyAstNodeTraverser.variable_matchers,
    )

    @classmethod
    def reset(cls):
        del cls._cache
        del cls.parser
        cls._cache = WorkbookState()
        cls.parser = RelParser(
        RelPyAstNodeTraverser(cls._cache),
        RelPyAstNodeTraverser.node_configuration,
        RelPyAstNodeTraverser.variable_matchers,
        )

    def __init__(self, file_path: str = None):
        self.file_path = file_path

    def expr(self, string: str):
        if re.search(r"\n|\r\n?", string):
            tasks = []
            lines = string.splitlines()
            for line in lines:
                if line.strip():
                    # If line starts with "#", ignore it
                    if line.startswith("#"):
                        continue

                    # If line contains "#", split and take the part before "#"
                    if "#" in line:
                        line = line.split("#")[0].strip()

                    if ";" in line:
                        lines = line.split(";")
                        for l in lines:
                            tasks.append(l.strip())
                        continue

                    tasks.append(line.strip())
            res = [Workbook.parser.exe(task) for task in tasks if task.strip()]
            return [r for r in res if r != None]

        return Workbook.parser.exe(string)

    def parse(self, string: str):
        return Workbook.parser.parse(string)

    def tokens(self, string: str):
        return Workbook.parser.tokenize(string)

    def exe(self, file_path: str):
        self.file_path = file_path
        with open(self.file_path, "r") as file:
            tasks = []
            for line in file:
                if line.strip():
                    # If line starts with "#", ignore it
                    if line.startswith("#"):
                        continue

                    # If line contains "#", split and take the part before "#"
                    if "#" in line:
                        line = line.split("#")[0].strip()

                    if ";" in line:
                        lines = line.split(";")
                        for l in lines:
                            tasks.append(l.strip())
                        continue

                    tasks.append(line.strip())

            res = [Workbook.parser.exe(task) for task in tasks if task.strip()]

            return [r for r in res if r != None]

    def exec_cell(self, tag_name):
        """
        This function searches for raw cells that have a specific tag, removes newline characters,
        and returns their content.

        :param tag_name: The tag to search for in the cell metadata.
        :param notebook_path: The path to the Jupyter notebook file.
        :return: A list of lists, each containing strings from the raw cells that have the given tag, with newlines removed.
        """
        # Initialize an empty list to store the contents of the cells that match the tag
        content_list = []

        # Read the notebook as a JSON file
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                notebook_data = json.load(f)
        except FileNotFoundError:
            print(f"File not found: {self.file_path}")
            return []
        except json.JSONDecodeError as e:
            print(f"Error reading the notebook file: {e}")
            return []

        # Iterate through the cells and collect the raw cells with the specified tag
        for cell in notebook_data["cells"]:
            if (
                cell["cell_type"] == "raw"
                and "tags" in cell["metadata"]
                and tag_name in cell["metadata"]["tags"]
            ):
                # Remove newlines from each string in the cell's source and add to the list
                content_without_newlines = [
                    line.replace("\n", "") for line in cell["source"]
                ]
                content_list = [i for i in content_without_newlines if i]

        tasks = []
        for line in content_list:
            if line.strip():
                # If line starts with "#", ignore it
                if line.startswith("#"):
                    continue

                # If line contains "#", split and take the part before "#"
                if "#" in line:
                    line = line.split("#")[0].strip()

                if ";" in line:
                    lines = line.split(";")
                    for l in lines:
                        tasks.append(l.strip())
                    continue

                tasks.append(line.strip())

        res = [Workbook.parser.exe(task) for task in tasks if task.strip()]

        return [r for r in res if r != None]

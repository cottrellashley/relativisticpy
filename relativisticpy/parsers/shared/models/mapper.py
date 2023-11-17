from typing import List, Dict, Any, Type


class Mappers:
    @staticmethod
    def map_from_list(lst: List[Dict[str, Any]], obj: Type) -> List:
        """
        This method is used to map a list of dictionaries to a list of objects.
        Each dictionary in the list will be used as keyword arguments to the constructor of the object type.

        Args:
            lst (List[Dict[str, Any]]): A list of dictionaries.
            obj (Type): The type of the objects to map to.

        Returns:
            List: A list of instantiated objects of the specified type.
        """
        return [obj(**item) for item in lst]

    @staticmethod
    def map_from_dict(dic: Dict[str, Any], obj: Type) -> Any:
        """
        This method is used to map a single dictionary to an object.
        The dictionary will be used as keyword arguments to the constructor of the object type.

        Args:
            dic (Dict[str, Any]): A dictionary.
            obj (Type): The type of the object to map to.

        Returns:
            Any: An instantiated object of the specified type.
        """
        return obj(**dic)

    @staticmethod
    def map_to_dict(obj: Any) -> Dict[str, Any]:
        """
        This method is used to map an object's properties to a dictionary.
        The object's __dict__ attribute is returned, which contains a dictionary of the object's attributes.

        Args:
            obj (Any): The object to map from.

        Returns:
            Dict[str, Any]: A dictionary representation of the object's attributes.
        """
        return obj.__dict__

    @staticmethod
    def map_list_to_dicts(lst: List[Any]) -> List[Dict[str, Any]]:
        """
        This method is used to map a list of objects to a list of dictionaries.
        Each object in the list is mapped to a dictionary using the object's __dict__ attribute.

        Args:
            lst (List[Any]): A list of objects.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries representing the objects' attributes.
        """
        return [obj.__dict__ for obj in lst]

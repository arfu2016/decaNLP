import csv
import json

import six


class Example(object):
    """Defines a single training or test example.

    Stores each column of the example as an attribute.
    本质上是把字典或者record转变成了类实例
    """

    @classmethod
    def fromJSON(cls, data, fields):
        return cls.fromdict(json.loads(data), fields)

    @classmethod
    def fromdict(cls, data, fields):
        ex = cls()
        for key, vals in fields.items():
            if key not in data:
                raise ValueError("Specified key {} was not found in "
                                 "the input data".format(key))
            if vals is not None:
                if not isinstance(vals, list):
                    vals = [vals]
                for val in vals:
                    name, field = val
                    setattr(ex, name, field.preprocess(data[key]))
        return ex

    @classmethod
    def fromTSV(cls, data, fields):
        return cls.fromlist(data.split('\t'), fields)

    @classmethod
    def fromCSV(cls, data, fields):
        data = data.rstrip("\n")
        # If Python 2, encode to utf-8 since CSV doesn't take unicode input
        if six.PY2:
            data = data.encode('utf-8')
        # Use Python CSV module to parse the CSV line
        parsed_csv_lines = csv.reader([data])

        # If Python 2, decode back to unicode (the original input format).
        if six.PY2:
            for line in parsed_csv_lines:
                parsed_csv_line = [six.text_type(col, 'utf-8') for col in line]
                break
        else:
            parsed_csv_line = list(parsed_csv_lines)[0]
        return cls.fromlist(parsed_csv_line, fields)

    @classmethod
    def fromlist(cls, data, fields):
        ex = cls()
        # 生成类的实例
        for (name, field), val in zip(fields, data):
            if field is not None:
                if isinstance(val, six.string_types):
                    val = val.rstrip('\n')
                setattr(ex, name, field.preprocess(val))
                # 要根据所在field来调用preprocess函数, 属性的名字也是由fields来决定的
        return ex
    # Example就是个很简单的类，并没有复杂的继承

    @classmethod
    def fromtree(cls, data, fields, subtrees=False):
        try:
            from nltk.tree import Tree
        except ImportError:
            print("Please install NLTK. "
                  "See the docs at http://nltk.org for more information.")
            raise
        tree = Tree.fromstring(data)
        if subtrees:
            return [cls.fromlist(
                [' '.join(t.leaves()), t.label()], fields) for t in tree.subtrees()]
        return cls.fromlist([' '.join(tree.leaves()), tree.label()], fields)

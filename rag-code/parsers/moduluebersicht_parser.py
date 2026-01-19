from pathlib import Path
from typing import Any, Dict, List
from haystack import Document
import numpy as np # type: ignore
from parsers.base_parser import BaseParser
import regex as re
class ModuluebersichtParser(BaseParser):
    doctypes = ["Moduluebersicht"]
    def parse(self, path: Path, meta: Dict[Any, Any]) -> List[Document]:
        assert path.exists() and path.is_file() and path.name.endswith(".md"), "Path must point to a .md file"
        lines = np.array([""])
        res: list[Document] = []
        with open(path, "r") as f:
            lines = f.read().splitlines()
        lines_without_images_or_course_name: list[str] = []
        i = 0
        start_lines_of_sections: list[int] = [] 
        for l in lines:
            if not "![]" in l:
                
                if re.match("#.*(Wintersemester|Sommersemester|Anmerkungen).*", l):
                    start_lines_of_sections.append(i)
                elif l.startswith("#"):
                    continue
                l_cells = [c.strip() for c in l.split("|")]
                lines_without_images_or_course_name.append("|".join(l_cells))
                i += 1
        sections: list[str] = []
        for i in range(i, len(start_lines_of_sections) - 1):
            start_idx, end_idx = start_lines_of_sections[i], start_lines_of_sections[i + 1]
            sections.append(
                "\n".join(lines_without_images_or_course_name[start_idx:end_idx])
            )
            
        for text in sections:
            res.append(Document(
                
            ))
        return res
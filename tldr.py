from tempfile import TemporaryDirectory
import arxiv
import tarfile
import re
from llama_cpp import Llama
def get_paper_summary(paper:arxiv.Result) -> str:
    with TemporaryDirectory() as tmpdirname:
        file = paper.download_source(dirpath=tmpdirname)
        with tarfile.open(file) as tar:
            tex_files = [f for f in tar.getnames() if f.endswith('.tex')]
            if len(tex_files) == 0:
                return None
            #read all tex files
            introduction = ""
            conclusion = ""
            for t in tex_files:
                f = tar.extractfile(t)
                content = f.read().decode('utf-8')
                #remove comments
                content = re.sub(r'%.*\n', '\n', content)
                content = re.sub(r'\\begin{comment}.*?\\end{comment}', '', content, flags=re.DOTALL)
                content = re.sub(r'\\iffalse.*?\\fi', '', content, flags=re.DOTALL)
                #remove redundant \n
                content = re.sub(r'\n+', '\n', content)
                #remove cite
                content = re.sub(r'~?\\cite.?\{.*?\}', '', content)
                #remove figure
                content = re.sub(r'\\begin\{figure\}.*?\\end\{figure\}', '', content, flags=re.DOTALL)
                #remove table
                content = re.sub(r'\\begin\{table\}.*?\\end\{table\}', '', content, flags=re.DOTALL)
                #find introduction and conclusion
                # end word can be \section or \end{document} or \bibliography or \appendix
                match = re.search(r'\\section\{Introduction\}.*?(\\section|\\end\{document\}|\\bibliography|\\appendix|$)', content, flags=re.DOTALL)
                if match:
                    introduction = match.group(0)
                match = re.search(r'\\section\{Conclusion\}.*?(\\section|\\end\{document\}|\\bibliography|\\appendix|$)', content, flags=re.DOTALL)
                if match:
                    conclusion = match.group(0)
                
    return introduction, conclusion

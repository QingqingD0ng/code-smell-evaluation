import lxml.html
from lxml import etree

def match_pubdate(node, pubdate_xpaths):
    for xpath in pubdate_xpaths:
        try:
            result = node.xpath(xpath)
            if result:
                return result[0]
        except etree.XPathEvalError:
            continue
    return None

# Example usage:
# parser = lxml.html.HTMLParser()
# tree = lxml.html.fromstring(html_content, parser=parser)
# pubdate = match_pubdate(tree, ['//span[@class="pubdate"]/text()', '//div[@class="article-date"]/text()'])
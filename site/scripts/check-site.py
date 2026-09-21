#!/usr/bin/env python3
"""Check built site navigation and links to files in this checkout."""
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urljoin, urlsplit

ROOT = Path(__file__).resolve().parents[2]
DIST = ROOT / 'site/dist'
BASE = '/VeriTile/'
ORIGIN = 'https://lizn-zn.github.io'

class Page(HTMLParser):
    def __init__(self, path):
        super().__init__()
        self.ids = set()
        self.links = []
        self.feed(path.read_text())
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if 'id' in attrs:
            self.ids.add(attrs['id'])
        if tag == 'a' and 'href' in attrs:
            self.links.append(attrs['href'])

def main():
    pages = {p: Page(p) for p in DIST.rglob('*.html')}
    if not pages:
        raise SystemExit('No built pages. Build the site first.')
    errors, checked = set(), 0
    for path, page in pages.items():
        relative = path.relative_to(DIST).as_posix()
        route = BASE + (relative[:-len('index.html')] if relative.endswith('index.html') else relative)
        for href in page.links:
            url = urlsplit(urljoin(ORIGIN + route, href))
            if url.netloc == 'github.com' and url.path.startswith('/Lizn-zn/VeriTile/'):
                parts = url.path.split('/')
                if len(parts) >= 6 and parts[3] in ('blob', 'tree') and parts[4] == 'main':
                    target = ROOT / unquote('/'.join(parts[5:]))
                    if not target.resolve().is_relative_to(ROOT) or not target.exists():
                        errors.add(f'{relative}: missing repository file {href}')
                    checked += 1
                continue
            if url.netloc != 'lizn-zn.github.io' or url.scheme not in ('http', 'https'):
                continue
            if not url.path.startswith(BASE):
                errors.add(f'{relative}: missing base prefix {href}')
                continue
            target = DIST / unquote(url.path.removeprefix(BASE))
            if target.is_dir():
                target /= 'index.html'
            if not target.exists():
                errors.add(f'{relative}: missing page {href}')
            elif url.fragment and target in pages and unquote(url.fragment) not in pages[target].ids:
                errors.add(f'{relative}: missing anchor {href}')
            checked += 1
    if errors:
        print('\n'.join(sorted(errors)))
        raise SystemExit(f'{len(errors)} broken links.')
    print(f'{len(pages)} pages; {checked} internal and repository links checked.')

if __name__ == '__main__':
    main()

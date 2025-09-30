#!/usr/bin/env python3
"""
Markdown to Clean Text Converter
Creates clean text versions of markdown files for easy copying
"""

import os
import re

def clean_markdown_to_text(content):
    """Convert markdown content to clean text"""
    # Remove markdown syntax
    content = re.sub(r'^#{1,6}\s+', '', content, flags=re.MULTILINE)  # Headers
    content = re.sub(r'\*\*(.*?)\*\*', r'\1', content)  # Bold
    content = re.sub(r'\*(.*?)\*', r'\1', content)      # Italic
    content = re.sub(r'`(.*?)`', r'\1', content)        # Inline code
    content = re.sub(r'```.*?```', '', content, flags=re.DOTALL)  # Code blocks
    content = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', content)  # Links
    content = re.sub(r'^[-*+]\s+', '• ', content, flags=re.MULTILINE)  # Bullets
    content = re.sub(r'^\d+\.\s+', '', content, flags=re.MULTILINE)  # Numbers
    
    # Remove excessive emojis and special characters
    content = re.sub(r'[📚🔧🎯🏛️🔄🏗️💻🚀📋⚙️📊💰🔍📉🛠️📁✅❌⚠️🆔💡🔮🎨📈🚗⛽🔋🛣️]', '', content)
    content = re.sub(r'[─│┌┐└┘├┤┬┴┼═║╔╗╚╝╠╣╦╩╬▶◀●○▼▲]', '', content)
    
    # Clean up spacing
    content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)  # Multiple newlines
    content = re.sub(r'^\s+', '', content, flags=re.MULTILINE)  # Leading whitespace
    
    return content.strip()

def convert_file_to_text(md_file, output_file):
    """Convert markdown file to clean text"""
    print(f"Converting {md_file} to {output_file}")
    
    try:
        with open(md_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        clean_content = clean_markdown_to_text(content)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(clean_content)
        
        print(f"✅ Successfully converted to {output_file}")
        
    except Exception as e:
        print(f"❌ Error converting {md_file}: {str(e)}")

def main():
    """Main function to convert all markdown files to text"""
    files_to_convert = [
        ('IMPLEMENTATION.md', 'IMPLEMENTATION.txt'),
        ('LITERATURE_REVIEW.md', 'LITERATURE_REVIEW.txt'),
        ('NOTES.md', 'NOTES.txt')
    ]
    
    print("🔄 Creating clean text versions for easy copying...")
    print("=" * 60)
    
    for md_file, txt_file in files_to_convert:
        if os.path.exists(md_file):
            convert_file_to_text(md_file, txt_file)
        else:
            print(f"⚠️ File not found: {md_file}")
    
    print("=" * 60)
    print("✅ Text conversion completed!")
    print("\n📄 Generated text files:")
    for _, txt_file in files_to_convert:
        if os.path.exists(txt_file):
            file_size = os.path.getsize(txt_file) / 1024  # KB
            print(f"  • {txt_file} ({file_size:.1f} KB)")

if __name__ == "__main__":
    main()
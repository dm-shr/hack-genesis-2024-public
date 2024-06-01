from pdf2image import convert_from_path
from transformers import AutoModelForObjectDetection
import torch
from torchvision import transforms
from transformers import TableTransformerForObjectDetection
import numpy as np
import csv
import easyocr
from tqdm.auto import tqdm
import fire

class MaxResize(object):
    def __init__(self, max_size=800):
        self.max_size = max_size

    def __call__(self, image):
        width, height = image.size
        current_max_size = max(width, height)
        scale = self.max_size / current_max_size
        resized_image = image.resize((int(round(scale*width)), int(round(scale*height))))

        return resized_image

    
# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


def outputs_to_objects(outputs, img_size, id2label):
    m = outputs.logits.softmax(-1).max(-1)
    pred_labels = list(m.indices.detach().cpu().numpy())[0]
    pred_scores = list(m.values.detach().cpu().numpy())[0]
    pred_bboxes = outputs['pred_boxes'].detach().cpu()[0]
    pred_bboxes = [elem.tolist() for elem in rescale_bboxes(pred_bboxes, img_size)]

    objects = []
    for label, score, bbox in zip(pred_labels, pred_scores, pred_bboxes):
        class_label = id2label[int(label)]
        if not class_label == 'no object':
            objects.append({'label': class_label, 'score': float(score),
                            'bbox': [float(elem) for elem in bbox]})

    return objects

def detect_objects_in_images(images, detection_transform, model, id2label, device):
    image_objects_dict = {}
    for index, image in enumerate(images):
        pixel_values = detection_transform(image).unsqueeze(0)
        pixel_values = pixel_values.to(device)

        with torch.no_grad():
            outputs = model(pixel_values)

        objects = outputs_to_objects(outputs, image.size, id2label)
        image_objects_dict[index] = objects
    return image_objects_dict


def objects_to_crops(img, tokens, objects, class_thresholds, padding=10):
    """
    Process the bounding boxes produced by the table detection model into
    cropped table images and cropped tokens.
    """

    table_crops = []
    for obj in objects:
        if obj['score'] < class_thresholds[obj['label']]:
            continue

        cropped_table = {}

        bbox = obj['bbox']
        bbox = [bbox[0]-padding, bbox[1]-padding, bbox[2]+padding, bbox[3]+padding]

        cropped_img = img.crop(bbox)

        table_tokens = [token for token in tokens if iob(token['bbox'], bbox) >= 0.5]
        for token in table_tokens:
            token['bbox'] = [token['bbox'][0]-bbox[0],
                             token['bbox'][1]-bbox[1],
                             token['bbox'][2]-bbox[0],
                             token['bbox'][3]-bbox[1]]

        # If table is predicted to be rotated, rotate cropped image and tokens/words:
        if obj['label'] == 'table rotated':
            cropped_img = cropped_img.rotate(270, expand=True)
            for token in table_tokens:
                bbox = token['bbox']
                bbox = [cropped_img.size[0]-bbox[3]-1,
                        bbox[0],
                        cropped_img.size[0]-bbox[1]-1,
                        bbox[2]]
                token['bbox'] = bbox

        cropped_table['image'] = cropped_img
        cropped_table['tokens'] = table_tokens

        table_crops.append(cropped_table)

    return table_crops


def crop_objects_from_images(images, detected_objects, detection_class_thresholds, crop_padding):
    cropped_images = {}
    
    for index, image in enumerate(images):
        objects = detected_objects.get(index, [])

        if not objects:
            continue
        
        tokens = [] 
        tables_crops = objects_to_crops(image, tokens, objects, detection_class_thresholds, padding=crop_padding)
        cropped_table = tables_crops[0]['image'].convert("RGB")
        
        cropped_images[index] = cropped_table
        
    return cropped_images

def process_cropped_images(cropped_images, structure_transform, structure_model, device):
    all_cells = {}

    for index, cropped_table in cropped_images.items():
        all_cells[index] = []
        
        pixel_values = structure_transform(cropped_table).unsqueeze(0)
        pixel_values = pixel_values.to(device)

        # forward pass
        with torch.no_grad():
            outputs = structure_model(pixel_values)

        structure_id2label = structure_model.config.id2label
        structure_id2label[len(structure_id2label)] = "no object"

        cells = outputs_to_objects(outputs, cropped_table.size, structure_id2label)
        all_cells[index].extend(cells)
    
    return all_cells

def get_cell_coordinates_by_row(table_data):
    # Extract rows and columns
    rows = [entry for entry in table_data if entry['label'] == 'table row']
    columns = [entry for entry in table_data if entry['label'] == 'table column']

    # Sort rows and columns by their Y and X coordinates, respectively
    rows.sort(key=lambda x: x['bbox'][1])
    columns.sort(key=lambda x: x['bbox'][0])

    # Function to find cell coordinates
    def find_cell_coordinates(row, column):
        cell_bbox = [column['bbox'][0], row['bbox'][1], column['bbox'][2], row['bbox'][3]]
        return cell_bbox

    # Generate cell coordinates and count cells in each row
    cell_coordinates = []

    for row in rows:
        row_cells = []
        for column in columns:
            cell_bbox = find_cell_coordinates(row, column)
            row_cells.append({'column': column['bbox'], 'cell': cell_bbox})

        # Sort cells in the row by X coordinate
        row_cells.sort(key=lambda x: x['column'][0])

        # Append row information to cell_coordinates
        cell_coordinates.append({'row': row['bbox'], 'cells': row_cells, 'cell_count': len(row_cells)})

    # Sort rows from top to bottom
    cell_coordinates.sort(key=lambda x: x['row'][1])

    return cell_coordinates

def apply_ocr(cell_coordinates, cropped_table, reader):
    # let's OCR row by row
    data = dict()
    max_num_columns = 0
    for idx, row in enumerate(tqdm(cell_coordinates)):
      row_text = []
      for cell in row["cells"]:
        # crop cell out of image
        cell_image = np.array(cropped_table.crop(cell["cell"]))
        # apply OCR
        result = reader.readtext(np.array(cell_image))
        if len(result) > 0:
          # print([x[1] for x in list(result)])
          text = " ".join([x[1] for x in result])
          row_text.append(text)

      if len(row_text) > max_num_columns:
          max_num_columns = len(row_text)

      data[idx] = row_text

    print("Max number of columns:", max_num_columns)

    # pad rows which don't have max_num_columns elements
    # to make sure all rows have the same number of columns
    for row, row_data in data.copy().items():
        if len(row_data) != max_num_columns:
          row_data = row_data + ["" for _ in range(max_num_columns - len(row_data))]
        data[row] = row_data

    return data


def save_all_tables(all_cells, cropped_images, reader, output_path):
    for index, cells in all_cells.items():
        cell_coordinates = get_cell_coordinates_by_row(cells)
        cropped_table = cropped_images.get(index)
        data = apply_ocr(cell_coordinates, cropped_table, reader)
        
        output_filename = f'{output_path}/table_page{index+1}.csv'
        with open(output_filename, 'w') as result_file:
            wr = csv.writer(result_file, dialect='excel')
            for row, row_text in data.items():
                wr.writerow(row_text)


def main(file_path, output_path='./'):
    images = convert_from_path(file_path)

    model = AutoModelForObjectDetection.from_pretrained("microsoft/table-transformer-detection", revision="no_timm")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # update id2label to include "no object"
    id2label = model.config.id2label
    id2label[len(model.config.id2label)] = "no object"

    detection_transform = transforms.Compose([
    MaxResize(800),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
    detected_objects = detect_objects_in_images(images, detection_transform, model, id2label, device)

    detection_class_thresholds = {
    "table": 0.5,
    "table rotated": 0.5,
    "no object": 10
}
    crop_padding = 100

    cropped_images = crop_objects_from_images(images, detected_objects, detection_class_thresholds, crop_padding)

    # new v1.1 checkpoints require no timm anymore
    structure_model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-structure-recognition-v1.1-all")
    structure_model.to(device)
    structure_transform = transforms.Compose([
    MaxResize(1000),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

    all_cells = process_cropped_images(cropped_images, structure_transform, structure_model, device)

    reader = easyocr.Reader(['en', 'ru']) 
    save_all_tables(all_cells, cropped_images, reader, output_path)

if __name__ == "__main__":
    fire.Fire(main)
    